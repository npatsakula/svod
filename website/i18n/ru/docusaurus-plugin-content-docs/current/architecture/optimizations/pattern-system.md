---
sidebar_label: Движок паттернов
sidebar_position: 0
---

# Движок паттернов

Почти каждый проход в Svod — это `graph_rewrite` поверх матчера, построенного макросом `patterns!`: стадии rangeify, символьный упроститель, expander, девекторизатор, декомпозиции, перемещение гейтов. Исключения — несколько обычных обходов графа (`memory_coalescing`, `merge_register_read_ends`, `linearize`) и построчная перезапись, работающая над линейным списком инструкций. Эта страница — справочник по макросу и движку. Исходники: `macros/src/patterns/`, `ir/src/pattern/`, `ir/src/rewrite/engine.rs`; аналоги в Tinygrad — `UPat` и `graph_rewrite` в `tinygrad/uop/ops.py`.

## DSL `patterns!`

Блок — это список правил `pattern [if guard] => body`, перед которым может стоять `@context Type;`. Левая часть — синтаксис паттернов Rust, расширенный тем, что паттерн Rust не может выразить сквозь ребро `Arc<UOp>`. Реальные правила из `schedule/src/symbolic/patterns.rs`:

```rust
// constant folding over thirteen binary ops, one rule body
for op in binary [Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr] {
    op(a @const(a_val), _b @const(b_val))
      => eval_binary_op(op, a_val, b_val).and_then(|r| folded_const(a.dtype(), r)),
},

// commutative identity with a guard; `name @ @zero` binds the constant node too
Add[x, zero @ @zero]
    if !x.dtype().is_float()
        || matches!(zero.op(), Op::Const(ConstValueHash(ConstValue::Float(v))) if v.is_sign_negative())
    => x.clone(),
Mul[x, @one] => x.clone(),

// a repeated name means the same node (Arc::ptr_eq), here across a struct field
original @ FloorDiv(x, x) => exact_integer_rewrite(original, 1.into_uop(x.dtype())),
original @ FloorMod(range @ Range { end, .. }, end) => exact_integer_rewrite(original, range.clone()),

// struct ops match by field; `..` skips the rest
Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
    if x.dtype() == *outer && can_safe_cast(outer, intermediate)
    => x.clone(),
```

Матчер с состоянием, из `schedule/src/expand.rs`:

```rust
crate::cached_patterns! {
    @context RangeMap;
    reduce @ Reduce { .. } => expand_reduce(reduce),
    range @ Range { end: _, axis_id, axis_type }
        if matches!(axis_type, AxisType::Upcast | AxisType::Unroll) && ctx.contains_key(axis_id)
        => expand_range(ctx, range),
    Wmma { a, b, c, metadata } if metadata.upcast_axes.is_some() => expand_wmma(ctx, a, b, c, metadata),
}
```

| Форма | Значение |
|------|---------|
| `Add(x, y)` | ALU-операция по виду, позиционная, упорядоченная. Имена разрешаются через `svod_ir::op::alu`, так что неверное имя или арность — ошибка компиляции. `UnaryOp`, `BinaryOp` (`Add, Mul, Sub, FloorMod, CMod, Max, Pow, FloorDiv, CDiv, Fdiv, Lt, Le, Eq, Ne, Gt, Ge, And, Or, Xor, Shl, Shr, Threefry`), `TernaryOp` (`Where`, `MulAcc`). |
| `Add[x, y]` | Коммутативная: ровно два потомка, пробуются оба порядка; guard и тело генерируются один раз и повторяются для каждого порядка. |
| `Cast { src: x, dtype }` | Структурная операция по полям. Поле является дочерним паттерном, если это `_`, `@..`, имя в snake_case или идентификатор, применённый к `(..)`/`[..]`/`{..}`/`@`; всё остальное — дословный паттерн Rust (`axis_type: AxisType::Upcast`, `index: 2`). `Some(pat)`/`None` сопоставляют потомков типа `Option<Arc<UOp>>` (`Load { alt: None, gate: Some(g), .. }`). Операции без полей пишутся голыми (`Noop`). |
| `x` / `_` / `name @ pattern` | привязать узел / игнорировать / привязать всё подсопоставление |
| `c @const(v)` | привязать узел `CONST` и его `ConstValue` |
| `c @vconst(vs)` / `c @anyconst(vs)` | линии `VCONST` / `CONST` или `VCONST` как `Vec<ConstValue>` |
| `Const(<rust pattern>)` | паттерн Rust над `ConstValue` |
| `@zero` / `@one` | скалярный `CONST` 0 / 1 любого числового dtype (`is_zero` также сопоставляет `-0.0` и `false`) |
| повторяющееся имя | тот же узел (`Arc::ptr_eq`); повторяющееся имя значения `@const` сравнивает значения |
| `for op in binary [A, B]` / `[*]` | одно тело правила для нескольких операций (или всех операций вида); `op` — значение операции во время выполнения, доступное в guard и теле |
| `pat if guard => body` | guard видит все привязки и `ctx` |
| `=> body` | `Arc<UOp>`, `Option<Arc<UOp>>` (`None` означает отказ) или `RewriteResult`; `?` работает, голая привязка возвращает клон |
| `@context Type;` | первый элемент; замыкание получает `ctx: &mut Type` |

`cached_patterns!` имеет ту же грамматику и возвращает `&'static TypedPatternMatcher<C>` из `LazyLock`; `patterns!` строит новый матчер. Оба реэкспортируются из `svod_schedule`.

### Что генерирует макрос

`Op` несёт `#[op_enum]`/`PatternEnum`, который генерирует `svod_ir::op::pattern_derived::OpKey` — один плотный индекс на вид операции, с отдельным слотом для каждой подоперации у сгруппированных `Unary`/`Binary`/`Ternary`, — и `OpMask`. Блок `patterns!` компилируется в **одно замыкание**, регистрируемое через `SimplifiedPatternMatcher::add_block`, плюс константную таблицу `(root mask, early-reject mask)` на каждое правило:

- Идущие подряд правила с одним константным видом корня делят `match __key { __KEY_Add => { .. } .. }`; внутри ветви правила сохраняют порядок исходника.
- Правила без константного корня — подстановочные (`x if ..`), блоки `for`, корни `@anyconst` — генерируются последовательными шагами *между* этими `match`, поэтому приоритет — чистый порядок исходника, а не «сначала индексированные, подстановочные последними».
- Каждое правило начинается с проверки раннего отказа: виды операций, которых требуют его фиксированные позиции потомков, образуют битовую маску, сверяемую с `src_ops` корня (`UPat.early_reject` в Tinygrad).
- Коммутативные места становятся лениво сцепленными итераторами кандидатов; вложенные коммутативные узлы — вложенными циклами; тело повторяется для каждого порядка.
- Блок `for` компилируется один раз на тело правила; переменная операции привязывается от корня во время выполнения.

`SimplifiedPatternMatcher<C>` (`TypedPatternMatcher<C = ()>` — псевдоним) — список сегментов, по одному на блок, у каждого корневая `OpMask` и замыкание. `rewrite(node, ctx)` просматривает сегменты, пропускает те, в чьей маске нет вида узла, и возвращает первый результат, отличный от `NoMatch`. `a + b` добавляет сегменты `b` после сегментов `a`, так что правила левого операнда выигрывают. `with_context::<D>()` поднимает `TypedPatternMatcher<()>` до матчера с контекстом `D` (принимает `&self`); написанные вручную замыкания добавляются через `add`, `add_rejecting`, `add_wildcard`. `Matcher<C>` — трейт (`fn rewrite(&self, &Arc<UOp>, &mut C) -> RewriteResult`); `DemoteFloat` в `late/dtype.rs` реализует его напрямую.

## Движок перезаписи

`ir/src/rewrite/engine.rs` — стековый порт `unified_rewrite` из Tinygrad. Каждый узел проходит три стадии:

| Стадия | Что происходит |
|-------|--------------|
| 0 — PushChildren | Если задан матчер `bpm`, применить его к этому узлу до неподвижной точки *до* спуска (паттерны видят исходных потомков). `Gate(node)` записывает замену и пропускает потомков. Затем положить на стек потомков, затем запись стадии 1 для этого узла. |
| 1 — ApplyPatterns | Разрешить потомков через карту замен (лист ожидания, если какой-то ещё не готов). Если потомок изменился, пересобрать узел и отправить пересобранный узел обратно на стадию 0. Иначе применить `pm`; результат `Rewritten` кладётся на стадию 0 — полностью заново обходится и сопоставляется, что и даёт неподвижную точку, — со ссылкой стадии 2. |
| 2 — Link | Отобразить исходный узел в итоговый результат его замены. |

Результаты мемоизируются по `UOp::id` (`replace`, `bpm_cache`; `Gate` никогда не кешируется). Два лимита: `REWRITE_STACK_LIMIT = 500_000` записей стека (`"infinite loop in graph_rewrite (stack too big: ..)"`) и множество `bpm_seen` на узел, вызывающее панику, когда неподвижная точка снизу вверх повторно посещает узел. Ограничения на число итераций нет.

| Точка входа | Матчеры |
|-------------|----------|
| `graph_rewrite(pm, root, ctx)` | `pm` на стадии 1 — правила видят переписанных потомков (по умолчанию в Tinygrad) |
| `graph_rewrite_bottom_up(bpm, root, ctx)` | `bpm` на стадии 0 — правила видят исходных потомков (`bottom_up=True` в Tinygrad); `Gate` учитывается |
| `graph_rewrite_with_bpm(pm, bpm, root, ctx)` | оба; используется только в тестах |
| `graph_rewrite_walk(bpm, root, ctx)` | один проход, замены повторно не обходятся (`walk=True` в Tinygrad) |
| варианты `*_preserve_calls` | то же самое, но без входа в тела `CALL`/`FUNCTION` и внутренности `PROGRAM` (`enter_calls=False` в Tinygrad) |

`RewriteResult` — это `NoMatch`, `Rewritten(Arc<UOp>)` или `Gate(Arc<UOp>)`; в матчере `pm` `Gate` трактуется как `NoMatch`. Разрез на ядра использует `Gate`, чтобы `split_all_stores` не спускался в уже сформированный `SINK` ядра (`rangeify/kernel.rs`). `debug_assert` срабатывает, если правило возвращает тот же узел, который получило.

Единственный драйвер не над графом — `line_rewrite` (`linearize/mod.rs`): он один раз проходит по линейному списку инструкций, позволяет каждой записи развернуться в несколько и подставляет последующие источники через карту. Его единственный клиент — `line_rewrite_cleanups`, развёртывание `STORE` с гейтом → `IF`/`STORE`/`ENDIF`.

`RUST_LOG=svod_ir::pattern=trace` логирует каждое сопоставление (`op_key`); он не логирует, какое правило сработало и какие пробовались.

## Композиция упорядочена

Матчеры компонуются через `+` в фиксированном порядке, и этот порядок несёт смысл. `symbolic_simple()` начинается с `propagate_invalid`, потому что иначе `x * 0 → 0` стёр бы `MUL(0, WHERE(c, x, Invalid))` вместе с его валидностью; `with_tier2` ставит канонизацию перед объединением слагаемых, а свёртку ALU — перед правилами сравнения, потому что каждая группа открывает сопоставления для следующей (см. [алгебраическое упрощение](./algebraic-simplification.md)). Добавить правило — значит выбрать, в каком месте этого порядка оно срабатывает.

## Проверка перезаписей с помощью Z3 {#verifying-rewrites-with-z3}

`schedule/src/z3/` (фича `z3`, опциональная зависимость `z3 = "0.21"`, системная `libz3`; её предоставляет nix flake) проверяет перезаписи вместо того, чтобы им доверять:

- `convert.rs` переводит дерево UOp в терм Z3: `CONST` (int, uint, bool; float и `Invalid` отвергаются), `DefineVar` как ограниченное целое, `RANGE` как свежая переменная с `0 <= r < end`, `Neg`, целочисленные бинарные операции `Add, Sub, Mul, FloorDiv, FloorMod, CDiv, CMod, Max, Lt, Eq, Ne` (`And`/`Or` на bool), `WHERE` и `MulAcc` на целых и `CAST` как свежая переменная, ограниченная dtype и связанная со своим источником, когда диапазон источника укладывается. `alu.rs` задаёт `CDiv`/`CMod` семантику усечения как в C; деление с округлением вниз строится поверх них. Всё остальное — `ConversionError`.
- `verify_equivalence(original, simplified)` переводит оба выражения в один контекст и утверждает `original != simplified`: `UNSAT` доказывает перезапись, `SAT` возвращает `CounterExample::Found { model, .. }`, таймаут — `Unknown`.

```rust
/// The identity elimination `x + 0 = x` is pointer-identical and Z3-proven.
#[test]
fn z3_verify_identity_add_zero(x in arb_var_uop(DType::Int32)) {
    let zero = UOp::native_const(0i32);
    let expr = x.try_add(&zero).expect("ADD accepts matching dtypes");
    let simplified = rewrite(Matchers::simple(), expr.clone());
    prop_assert!(Arc::ptr_eq(&simplified, &x));
    verify_equivalence(&expr, &simplified).expect("Z3 should verify x + 0 = x");
}
```

Что покрыто (`schedule/src/test/`): написанные вручную строки через `symbolic_simple` (`unit/z3/symbolic_patterns.rs`), proptest-оракулы над `arb_arithmetic_tree_bounded_up_to` и `arb_known_property_graph` через `symbolic_simple` и `symbolic` (`property/oracles.rs`, по 300–500 случаев каждый) и двойной прогон структурных символьных тестов, который перепроверяет каждую строку с помощью Z3, если она переводится (`unit/symbolic/mod.rs`). Тест проваливает только `Found`; `Unknown` и `ConversionFailed` допускаются, а тест живости следит за тем, что арифметическое ядро по-прежнему переводится. Запуск: `cargo test -p svod-schedule --features z3,proptest`; CI запускает те же фичи через `nix flake check`.

Доказательство проводится над неограниченными целыми на выборке выражений из ограниченного подмножества операций — это сильная сеть против регрессий для упростителя индексов, а не верификация каждого паттерна.
