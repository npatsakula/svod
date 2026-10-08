---
sidebar_label: Первое ядро
---

# Первое ядро {#your-first-kernel}

Ядро tk3 — это функция на Rust, которая записывает тайловую программу билдером `Kernel`.
Каждый вызов добавляет одну инструкцию в открытый блок и возвращает типизированный дескриптор.
Дескриптор несёт свой уровень памяти (`Gmem<T>`, `Shared<T>`, `Regs<T>`) и тип элемента (`BF16`,
`F16`, `F32`, `I32`, `Bool`). Формы проверяются в момент записи инструкции. Скаляры — индексы
блоков, смещения, число итераций — это выражения `Sc`, собранные обычными операторами.

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
    let v = k.binary(v, two, BinaryOp::Mul);
    let v = k.binary(v, one, BinaryOp::Add);
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

## GEMM шаг за шагом {#the-gemm-step-by-step}

`kernels/gemm.rs` вычисляет `c = act(a·bᵀ + bias) + residual` (или `act(gate)·up` для gated-весов)
примерно в 180 строках вместе с типами спецификации и конфигурации. Фрагменты ниже приведены
дословно.

**Параметры и сетка.** Буферы плоские и рассчитаны на ёмкость батча. Связанная переменная
батча становится осью z сетки.

```rust
let a = k.param::<T>("a", ParamKind::In, cap * m * kk);
let b = k.param::<T>("b", ParamKind::In, halves * n * kk);
let bias = epi.bias.then(|| k.param::<T>("bias", ParamKind::In, halves * n));
let residual = epi.residual.then(|| k.param::<T>("residual", ParamKind::In, cap * m * n));
let c = k.param::<T>("c", ParamKind::Out, cap * m * n);
let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
let (gz, bb) = batch.axis(&mut k);
k.grid([Sc::from(gm), Sc::from(gn), gz]);
k.warps(cfg.warps[0] * cfg.warps[1]);
```

**View с границами вместо паддинга.** `bound(len, tile)` равно `Some(len)` только там, где сетка
тайлов выходит за `len`. Любые `m` и `n` работают без копирования. Кратным `bk` обязано быть
только `k`.

```rust
let (bx, by) = tile_order(&mut k, gm, gn, cfg.group_m);
let (row0, col0) = (bx * bm, by * bn);
let (m_bound, n_bound) = (bound(m, bm), bound(n, bn));
let a_view = k.view(a, batch_offset(&bb, m * kk), [kk, 1], Shape::new(bm, bk), [m_bound.clone(), None]);
let a_view = k.at(a_view, row0.clone(), 0);
```

`tile_order` обходит группы по `group_m` строк тайлов, чтобы одновременно резидентные блоки
делили B в L2.

**Конвейер.** `pipeline(extent, stages, init, produce, consume)` объявляет производителя и
потребителя над кольцом из `stages` слотов разделяемой памяти. Автор говорит, что копировать в
слот и что вычислять из него. [Шаблон расписания](./layouts-and-lowering#schedule-templates)
решает, насколько вперёд идут копии, и расставляет каждое ожидание и барьер.

```rust
k.pipeline(
    trips,
    stages,
    init,
    |k, step, slot| {
        let koff = step * bk;
        for (src, alloc, shape) in
            std::iter::once((a, a_s, sa)).chain(bs.into_iter().zip(b_s).map(|(b, s)| (b, s, sb)))
        {
            let g = k.at(src, 0, koff.clone());
            let t = k.smem_slot::<T>(alloc, slot.clone(), shape);
            k.stage(t, g, CopyMode::Async);
        }
    },
    |k, _step, slot, accs| {
        let a_t = k.smem_slot::<T>(a_s, slot.clone(), sa);
        let mut i = 0;
        accs.map(|acc| {
            let b_t = k.smem_slot::<T>(b_s[i], slot.clone(), sb);
            i += 1;
            k.mma(acc, a_t, false, b_t, true)
        })
    },
)
```

`mma(acc, a, a_t, b, b_t)` — это `acc + A·B` на аккумуляторах f32. Инструкцию оно не называет:
понижение выбирает атом матричного ядра цели и даёт `a`, `b` и `acc` его раскладки.

**Эпилог** выполняется на аккумуляторе f32, а результат округляется один раз при записи.
Смещение — вектор-строка `[1, bn]`, транслируемый на тайл, а residual — полный тайл.

```rust
if let Some(residual) = residual {
    let r = tile(&mut k, residual);
    let r = load_f32(&mut k, r);
    out = k.binary(out, r, BinaryOp::Add);
}
let out = k.cast::<F32, T>(out);
let c_view = tile(&mut k, c);
k.store(c_view, out);
k.finish()
```

## Запуск на устройстве {#running-it-on-the-device}

`launch::graph_launch` принимает по одному тензору на каждый объявленный параметр, по порядку,
и возвращает первый выход как ленивый `Tensor`. Ядро выполняется при реализации результата,
как любое ядро графа.

```rust
use svod_dtype::{DType, default_device::default_device};
use svod_tensor::Tensor;
use svod_tk3::atoms::Target;
use svod_tk3::build::BF16;
use svod_tk3::kernels::Batch;
use svod_tk3::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use svod_tk3::launch::graph_launch;

let (m, n, k) = (1000, 512, 256);
let target = Target::for_device(&default_device()).expect("a GPU target");
let cfg = GemmCfg { tile: [64, 64, 32], stages: 3, warps: [2, 2], group_m: 8, unroll: true };
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &cfg.lowering(target), &[&a, &b, &c])?;
c.realize()?;
```

Модели так не делают: [`ops::linear`](./op-layer) выбирает конфигурацию, формирует выход и
откатывается на граф, когда ядро неприменимо.

## Вызовы билдера {#builder-calls}

| Группа | Вызовы |
|---|---|
| Объявления | `param`, `var` (переменная запуска, связываемая по имени), `grid`, `warps`, `smem` |
| Скаляры | `block(axis)`, `warp()`, `load_scalar(param, index)`, операторы `Sc` и `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| View | `view`, `at`, `smem_view`, `smem_slot` |
| Тайловые операции | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (вектор-строка или вектор-столбец транслируется), `compare`, `cast`, `where_`, `reduce`, `mma` |
| Перемещение | `stage` (global → shared, `CopyMode::Async` или `Sync`), `load`, `store` |
| Управление | `loop_` (переносимые регистровые тайлы), `pipeline`, `if_`, `select_if` (ветви, производящие тайлы) |

`role_block`, `raw` и `transpose` можно записать, но они пока не понижаются: понижение
возвращает для них `Error::Unsupported`.
