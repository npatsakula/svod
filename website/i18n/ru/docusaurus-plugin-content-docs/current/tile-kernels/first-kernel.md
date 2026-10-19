---
sidebar_label: Первое ядро
---

# Первое ядро {#your-first-kernel}

Ядро tk3 — это функция на Rust, которая записывает тайловую программу билдером `Kernel`.
Каждый вызов добавляет одну инструкцию в открытый блок и возвращает типизированный дескриптор.
Дескриптор несёт свой уровень памяти (`Gmem<T>`, `Shared<T>`, `Regs<T>`) и тип элемента (`BF16`,
`F16`, `F32`, `I32`, `Bool`); выделение разделяемой памяти — это `SmemRef<T>`. Формы проверяются
в момент записи инструкции. Скаляры — индексы блоков, смещения, число итераций — это выражения
`Sc`, собранные обычными операторами.

## Минимальное ядро {#a-minimal-kernel}

`y = 2·x + 1` над матрицей f32 `[rows, 64]`, по 16 строк на блок. Граница строк у view
заставляет строки последнего блока за пределами `rows` читаться как ноль и отбрасывает их
запись, поэтому `rows` не обязано быть кратным 16.

```rust
use svod_tk3::build::*;
use svod_tk3::interp::run;
use svod_tk3::ir::*;

fn axpb(rows: usize) -> Program {
    let cols = 64;
    let mut k = Kernel::new("axpb");
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let y = k.param::<F32>("y", ParamKind::Out, rows * cols);
    k.grid([Sc::from(rows.div_ceil(16)), Sc::from(1), Sc::from(1)]);
    k.warps(4);
    let row0 = k.block(0) * 16;
    let tile = |k: &mut Kernel, p: ParamRef<F32>| {
        let v = k.view(p, 0, [cols, 1], Shape::new(16, cols), [Some(Sc::from(rows)), None]);
        k.at(v, row0.clone(), 0)
    };
    let xv = tile(&mut k, x);
    let v = k.load(xv);
    let two = k.fill::<F32>(Shape::new(16, cols), Const::Float(2.0));
    let one = k.fill::<F32>(Shape::new(16, cols), Const::Float(1.0));
    let v = k.mul_add(v, two, one);
    let yv = tile(&mut k, y);
    k.store(yv, v);
    k.finish()
}

// On the host, no GPU: one Vec<f64> per parameter in, every parameter out.
let x: Vec<f64> = (0..40 * 64).map(f64::from).collect();
let out = run(&axpb(40), vec![x, vec![0.0; 40 * 64]], &[])?;
assert_eq!(out[1][64 * 39 + 1], 2.0 * (64.0 * 39.0 + 1.0) + 1.0);
```

`view(param, offset, [row_stride, col_stride], shape, bounds)` — окно в параметр.
`at(view, rows, cols)` сдвигает его, сохраняя границы относительно нового начала.

## GEMM {#a-gemm}

`c = a·bᵀ` над bf16 `a [m, k]` и `b [n, k]` с накоплением в f32. Это упрощённая форма
`kernels/gemm.rs`, где добавлены эпилоги (смещение, активация, gated-веса, масштаб, residual,
выход в f32), групповой порядок тайлов, разбиение редукции, ядро на уровне варп-группы и
ping-pong двух бригад.

```rust
fn gemm_nt(m: usize, n: usize, kk: usize, [bm, bn, bk]: [usize; 3], stages: usize) -> Program {
    assert!(kk.is_multiple_of(bk), "k is a multiple of bk");
    let mut k = Kernel::new("gemm_nt");
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, n * kk);
    let c = k.param::<BF16>("c", ParamKind::Out, m * n);
    k.grid([Sc::from(m.div_ceil(bm)), Sc::from(n.div_ceil(bn)), Sc::from(1)]);
    k.warps(4);
    let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
    let bound = |len: usize, tile: usize| (!len.is_multiple_of(tile)).then(|| Sc::from(len));
    let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [bound(m, bm), None]);
    let a_view = k.at(a_view, row0.clone(), 0);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [bound(n, bn), None]);
    let b_view = k.at(b_view, col0.clone(), 0);
    let (sa, sb) = (Shape::new(bm, bk), Shape::new(bn, bk));
    let a_s = k.smem::<BF16>("a_s", stages * sa.elems());
    let b_s = k.smem::<BF16>("b_s", stages * sb.elems());
    let acc = k.zeros::<F32>(Shape::new(bm, bn));
    let [acc] = k.pipeline(
        kk / bk,
        stages,
        [acc],
        |k, step, slot| {
            let koff = step * bk;
            let a_g = k.at(a_view, 0, koff.clone());
            let a_t = k.smem_slot(a_s, slot.clone(), sa);
            k.stage(a_t, a_g, CopyMode::Async);
            let b_g = k.at(b_view, 0, koff);
            let b_t = k.smem_slot(b_s, slot, sb);
            k.stage(b_t, b_g, CopyMode::Async);
        },
        |k, _step, slot, [acc]| {
            let a_t = k.smem_slot(a_s, slot.clone(), sa);
            let b_t = k.smem_slot(b_s, slot, sb);
            [k.mma(acc, a_t, false, b_t, true)]
        },
    );
    let out = k.cast::<F32, BF16>(acc);
    let c_view = k.view(c, 0, [n, 1], Shape::new(bm, bn), [bound(m, bm), bound(n, bn)]);
    let c_view = k.at(c_view, row0, col0);
    k.store(c_view, out);
    k.finish()
}
```

**Границы вместо паддинга.** `bound(len, tile)` равно `Some(len)` только там, где сетка тайлов
выходит за `len`, поэтому любые `m` и `n` работают без копирования. Кратным `bk` обязано быть
только `k`.

**Конвейер.** `pipeline(extent, stages, init, produce, consume)` объявляет производителя и
потребителя над кольцом из `stages` слотов разделяемой памяти и проносит `init` (здесь —
аккумулятор) через потребителя. Автор говорит, что копировать в слот и что вычислять из него.
[Шаблон расписания](./layouts-and-lowering#schedule-templates) решает, насколько вперёд идут
копии, и расставляет каждое ожидание и барьер.

**Произведение.** `mma(acc, a, a_t, b, b_t)` — это `acc + A·B` на аккумуляторах f32, здесь `B`
читается транспонированной. Инструкцию оно никогда не называет: понижение выбирает атом
матричного ядра цели и даёт `a`, `b` и `acc` его раскладки. Результат округляется один раз — при
приведении типа перед записью.

## Запуск на устройстве {#running-it-on-the-device}

`launch::graph_launch` принимает по одному тензору на каждый объявленный параметр, по порядку,
и возвращает первый выход как ленивый `Tensor`. Ядро выполняется при реализации результата,
как любое ядро графа. Библиотечный GEMM принимает `GemmCfg`, чьему `lowering` нужны измеренные
стоимости устройства.

```rust
use svod_dtype::{DType, default_device::default_device};
use svod_tensor::Tensor;
use svod_tk3::atoms::{Core, Device, Target};
use svod_tk3::build::BF16;
use svod_tk3::kernels::Batch;
use svod_tk3::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use svod_tk3::launch::graph_launch;

let (m, n, k) = (1000, 512, 256);
let target = Target::for_device(&default_device()).expect("a GPU target");
let device = Device { costs: svod_tk3::probe::costs(&target)?.costs, target };
let cfg =
    GemmCfg { tile: [64, 64, 32], stages: 2, warps: [2, 2], group_m: 8, unroll: true, core: Core::Warp, crews: 1 };
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &cfg.lowering(&device), &[&a, &b, &c])?;
c.realize()?;
```

Модели так не делают: [`ops::linear`](./op-layer) выбирает конфигурацию, формирует выход и
откатывается на граф, когда ядро неприменимо.

## Вызовы билдера {#builder-calls}

| Группа | Вызовы |
|---|---|
| Объявления | `param`, `var` (переменная запуска, связываемая по имени), `grid`, `warps`, `smem` |
| Скаляры | `block(axis)`, `warp()`, `row()` (внутри отображения gather), `load_scalar(param, index)`, операторы `Sc` и `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| View | `view`, `gather` (view с отображением строк), `at`, `narrow`, `smem_view`, `smem_view_as`, `smem_slot` |
| Тайловые операции | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (вектор-строка или вектор-столбец транслируется), `scale`, `compare` (`CmpOp`), `cast`, `cast_toward_zero`, `where_`, `mul_add` (одно округление), `reduce`, `reduce_f32`, `transpose`, `mma`, `mma_oriented`, `pin` |
| Перемещение | `stage` (global → shared, `CopyMode::Async` или `Sync`), `load`, `load_as_fill`, `store` |
| Управление | `loop_` (переносимые регистровые тайлы), `pipeline`, `if_`, `select_if` (ветви, производящие тайлы), `each_warp` (инструкция `Warps`), `barrier` |
