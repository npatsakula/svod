---
sidebar_label: Введение
---

# Svod

> **Альфа-версия.** Основная функциональность протестирована, но API нестабильны и могут измениться без предупреждения.

ML-компилятор на Rust, вдохновлённый [Tinygrad](https://github.com/tinygrad/tinygrad). Ленивые тензорные вычисления на основе UOp IR, паттерн-ориентированные оптимизации и кодогенерация под несколько бэкендов.

## Основные возможности

| Возможность | Описание |
|-------------|----------|
| **Декларативные оптимизации** | DSL `patterns!` для перезаписи графов с Z3-верифицированной корректностью |
| **Ленивые вычисления** | Тензоры строят граф вычислений, компиляция происходит только при `realize()` |
| **Отслеживание происхождения** | скоупы привязывают каждое ядро к пути модуля, месту вызова или узлу ONNX |
| **80+ IR-операций** | Арифметика, память, control flow, WMMA tensor cores |
| **20+ оптимизаций** | Свёртка констант, tensor cores, векторизация, развёртка циклов |

Подробнее об архитектуре — на [сайте документации](https://npatsakula.github.io/svod/).

## Структура проекта

| Крейт | Описание |
|-------|----------|
| [dtype](https://github.com/npatsakula/svod/tree/main/dtype/) | Система типов: скаляры, векторы, указатели, изображения |
| [macros](https://github.com/npatsakula/svod/tree/main/macros/) | Процедурные макросы (DSL `patterns!`) |
| [ir](https://github.com/npatsakula/svod/tree/main/ir/) | UOp-граф IR: 80+ операций, символьные целые, происхождение ядер |
| [device](https://github.com/npatsakula/svod/tree/main/device/) | Управление буферами: ленивое выделение, zero-copy view, LRU-кэширование |
| [schedule](https://github.com/npatsakula/svod/tree/main/schedule/) | Движок оптимизаций: 20+ проходов, RANGEIFY, Z3-верификация |
| [codegen](https://github.com/npatsakula/svod/tree/main/codegen/) | Кодогенерация: Clang (по умолчанию), LLVM JIT |
| [runtime](https://github.com/npatsakula/svod/tree/main/runtime/) | JIT-компиляция и выполнение ядер |
| [tensor](https://github.com/npatsakula/svod/tree/main/tensor/) | Высокоуровневый API ленивых тензоров |
| [onnx](https://github.com/npatsakula/svod/tree/main/onnx/) | Импортер ONNX-моделей |
| [arch](https://github.com/npatsakula/svod/tree/main/arch/) | Примитивы инференса |
| [model](https://github.com/npatsakula/svod/tree/main/model/) | Предобученные модели |

## Быстрый пример

```rust
use svod_tensor::Tensor;
use ndarray::array;

// Zero-copy from ndarray (C-contiguous fast path)
let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

// Lazy — nothing executes yet
let mut c = &a + &b;
c.realize()?;

// Zero-copy view into the result
let view = c.array_view::<f32>()?;
assert_eq!(view, array![[6.0, 8.0], [10.0, 12.0]].into_dyn());
```

## Пример DSL паттернов

```rust
use svod_schedule::patterns;

let optimizer = patterns! {
    // Identity folding (commutative)
    Add[x, @zero] => x,
    Mul[x, @one] => x,

    // Constant folding
    for op in binary [Add, Mul, Sub] {
        op(a @const(av), _b @const(bv))
          => eval_binary_op(op, av, bv).map(|r| UOp::const_(a.dtype(), r)),
    },
};
```

## Разработка

### Настройка окружения

#### Nix

Проект содержит предварительно настроенное Nix-окружение для разработки
со всеми зависимостями и компиляторами. Та же инфраструктура
используется для CI/CD, поэтому это предпочтительный способ разработки и тестирования.

```bash
nix develop # Open development shell
nix flake check # Run CI tests
nix fmt # Format source files
```

#### Установка вручную

| Зависимость | Версия | Обязательна | Описание |
|-------------|--------|-------------|----------|
| Rust | 1.85+ | да | Edition 2024 |
| LLVM | 21.x | да | Backend кодогенерации для CPU |
| Clang | - | да | C-компилятор для сборки LLVM |
| pkgconf | - | да | Инструмент конфигурации сборки |
| protobuf | - | да | Компиляция ONNX proto |
| zlib | >=1.3 | да | Библиотека сжатия |
| libffi | >=3.4 | да | Foreign function interface |
| libxml2 | >=2.13 | да | Парсинг XML |
| Z3 | >=4.15 | нет | SMT-решатель для верификации оптимизаций |
| Драйвер NVIDIA | CUDA >=12.8 (R570) | нет | `libcuda.so.1` для CUDA-бэкенда, загружается во время выполнения; тулкит не нужен |
| Таргеты Clang NVPTX / AMDGPU | - | нет | Компиляция GPU-ядер (`clang --print-targets`) |

## Тестирование

```bash
cargo test
cargo test --features z3,proptest  # With Z3 verification and PB generated tests
```
