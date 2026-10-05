---
sidebar_label: 模式引擎
sidebar_position: 0
---

# 模式引擎

Svod 中几乎每个 pass 都是基于 `patterns!` 宏构建的匹配器的一次 `graph_rewrite`：rangeify 各阶段、符号化简器、expander、devectorizer、各种分解、gate 移动。例外是少数普通的图遍历（`memory_coalescing`、`merge_register_read_ends`、`linearize`）以及在线性指令列表上运行的行重写。本页是该宏和引擎的参考。源码：`macros/src/patterns/`、`ir/src/pattern/`、`ir/src/rewrite/engine.rs`；Tinygrad 中的对应物是 `tinygrad/uop/ops.py` 里的 `UPat` 和 `graph_rewrite`。

## `patterns!` DSL

一个块是由 `pattern [if guard] => body` 规则组成的列表，前面可以有 `@context Type;`。左侧是 Rust 模式语法，并扩展了 Rust 模式无法跨越 `Arc<UOp>` 边表达的内容。以下是 `schedule/src/symbolic/patterns.rs` 中的真实规则：

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

一个有状态的匹配器，来自 `schedule/src/expand.rs`：

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

| 形式 | 含义 |
|------|---------|
| `Add(x, y)` | 按种类匹配 ALU 算子，位置参数，有序。名称经由 `svod_ir::op::alu` 解析，因此名称或元数错误会导致编译错误。`UnaryOp`、`BinaryOp`（`Add, Mul, Sub, FloorMod, CMod, Max, Pow, FloorDiv, CDiv, Fdiv, Lt, Le, Eq, Ne, Gt, Ge, And, Or, Xor, Shl, Shr, Threefry`）、`TernaryOp`（`Where`、`MulAcc`）。 |
| `Add[x, y]` | 可交换：恰好两个子节点，两种顺序都会尝试；guard 和 body 只生成一次，按每种顺序重试。 |
| `Cast { src: x, dtype }` | 按字段匹配结构体算子。当字段为 `_`、`@..`、snake_case 名称，或作用于 `(..)`/`[..]`/`{..}`/`@` 的标识符时，它是一个子模式；其他一律作为原样的 Rust 模式（`axis_type: AxisType::Upcast`、`index: 2`）。`Some(pat)`/`None` 匹配 `Option<Arc<UOp>>` 子节点（`Load { alt: None, gate: Some(g), .. }`）。单元算子直接写名称（`Noop`）。 |
| `x` / `_` / `name @ pattern` | 绑定节点 / 忽略 / 绑定整个子匹配 |
| `c @const(v)` | 绑定一个 `CONST` 节点及其 `ConstValue` |
| `c @vconst(vs)` / `c @anyconst(vs)` | `VCONST` 的各 lane / 以 `Vec<ConstValue>` 形式匹配 `CONST` 或 `VCONST` |
| `Const(<rust pattern>)` | 作用于 `ConstValue` 的 Rust 模式 |
| `@zero` / `@one` | 任意数值 dtype 的标量 `CONST` 0 / 1（`is_zero` 也匹配 `-0.0` 和 `false`） |
| 重复的名称 | 同一个节点（`Arc::ptr_eq`）；重复的 `@const` 值名称则比较值 |
| `for op in binary [A, B]` / `[*]` | 多个算子（或某一类的全部算子）共用一个规则体；`op` 是运行时的算子值，可在 guard 和 body 中使用 |
| `pat if guard => body` | guard 能看到所有绑定和 `ctx` |
| `=> body` | `Arc<UOp>`、`Option<Arc<UOp>>`（`None` 表示放弃）或 `RewriteResult`；可以使用 `?`，单独一个绑定会返回其克隆 |
| `@context Type;` | 第一项；闭包接收 `ctx: &mut Type` |

`cached_patterns!` 语法相同，从 `LazyLock` 返回 `&'static TypedPatternMatcher<C>`；`patterns!` 则构建一个新的匹配器。两者都从 `svod_schedule` 重新导出。

### 宏生成了什么

`Op` 带有 `#[op_enum]`/`PatternEnum`，它生成 `svod_ir::op::pattern_derived::OpKey` —— 每种算子一个稠密索引，分组的 `Unary`/`Binary`/`Ternary` 每个子算子一个槽位 —— 以及 `OpMask`。一个 `patterns!` 块编译为通过 `SimplifiedPatternMatcher::add_block` 注册的**一个闭包**，外加每条规则一项 `(root mask, early-reject mask)` 的常量表：

- 根节点种类为同一个常量的连续规则共享一个 `match __key { __KEY_Add => { .. } .. }`；在同一分支内，规则保持源码顺序。
- 没有常量根的规则 —— 通配符（`x if ..`）、`for` 块、以 `@anyconst` 为根的规则 —— 作为顺序步骤生成在这些 `match` *之间*，因此优先级完全取决于源码顺序，而不是“先索引规则、后通配规则”。
- 每条规则以一个提前拒绝测试开始：其固定子节点位置所要求的算子种类构成一个位掩码，与根节点的 `src_ops` 比较（即 Tinygrad 的 `UPat.early_reject`）。
- 可交换位置变成惰性链接的候选迭代器；嵌套的可交换节点变成嵌套循环；body 按每种顺序重试。
- `for` 块对每个规则体只编译一次；算子变量在运行时从根节点绑定。

`SimplifiedPatternMatcher<C>`（别名为 `TypedPatternMatcher<C = ()>`）是一个段列表，每个块一段，每段带有根 `OpMask` 和闭包。`rewrite(node, ctx)` 扫描各段，跳过掩码不含该节点种类的段，返回第一个非 `NoMatch` 的结果。`a + b` 把 `b` 的段追加在 `a` 的段之后，因此左操作数的规则优先。`with_context::<D>()` 把 `TypedPatternMatcher<()>` 提升为 `D` 上下文的匹配器（它接受 `&self`）；手写闭包通过 `add`、`add_rejecting`、`add_wildcard` 加入。`Matcher<C>` 是 trait（`fn rewrite(&self, &Arc<UOp>, &mut C) -> RewriteResult`）；`late/dtype.rs` 中的 `DemoteFloat` 直接实现了它。

## 重写引擎

`ir/src/rewrite/engine.rs` 是 Tinygrad `unified_rewrite` 的基于栈的移植。每个节点经历三个阶段：

| 阶段 | 发生什么 |
|-------|--------------|
| 0 —— PushChildren | 若给定了 `bpm` 匹配器，在下降*之前*对该节点应用它直到不动点（模式看到的是原始子节点）。`Gate(node)` 记录一个替换并跳过子节点。然后压入子节点，再压入该节点的阶段 1 条目。 |
| 1 —— ApplyPatterns | 通过替换映射解析子节点（若某个子节点尚未就绪则进入等待列表）。若某个子节点发生了变化，则重建该节点并把重建后的节点送回阶段 0。否则应用 `pm`；`Rewritten` 结果以阶段 0 压栈 —— 完整地重新遍历和重新匹配，这就是不动点 —— 并附带一个阶段 2 链接。 |
| 2 —— Link | 把原始节点映射到其替换的最终结果。 |

结果按 `UOp::id` 记忆化（`replace`、`bpm_cache`；`Gate` 从不缓存）。有两个限制：`REWRITE_STACK_LIMIT = 500_000` 个栈条目（`"infinite loop in graph_rewrite (stack too big: ..)"`），以及一个逐节点的 `bpm_seen` 集合，当自底向上的不动点再次访问某个节点时会 panic。没有迭代次数上限。

| 入口 | 匹配器 |
|-------------|----------|
| `graph_rewrite(pm, root, ctx)` | `pm` 在阶段 1 —— 规则看到的是重写后的子节点（Tinygrad 默认） |
| `graph_rewrite_bottom_up(bpm, root, ctx)` | `bpm` 在阶段 0 —— 规则看到的是原始子节点（Tinygrad `bottom_up=True`）；会遵从 `Gate` |
| `graph_rewrite_with_bpm(pm, bpm, root, ctx)` | 两者皆有；仅测试使用 |
| `graph_rewrite_walk(bpm, root, ctx)` | 单次遍历，替换结果不再遍历（Tinygrad `walk=True`） |
| `*_preserve_calls` 变体 | 同上，但不进入 `CALL`/`FUNCTION` 函数体或 `PROGRAM` 内部（Tinygrad `enter_calls=False`） |

`RewriteResult` 是 `NoMatch`、`Rewritten(Arc<UOp>)` 或 `Gate(Arc<UOp>)`；在 `pm` 匹配器中，`Gate` 被视为 `NoMatch`。内核切分用 `Gate` 阻止 `split_all_stores` 深入已经成形的内核 `SINK`（`rangeify/kernel.rs`）。若某条规则返回了传给它的同一个节点，会触发一个 `debug_assert`。

唯一的非图驱动器是 `line_rewrite`（`linearize/mod.rs`）：它遍历一次线性指令列表，允许每个条目展开为多个，并通过映射替换后续的源。它唯一的使用者是 `line_rewrite_cleanups`，即带 gate 的 `STORE` → `IF`/`STORE`/`ENDIF` 展开。

`RUST_LOG=svod_ir::pattern=trace` 记录每次匹配（`op_key`）；它不会记录触发的是哪条规则，也不记录尝试过哪些规则。

## 组合是有序的

匹配器通过 `+` 以固定顺序组合，顺序本身具有意义。`symbolic_simple()` 以 `propagate_invalid` 开头，否则 `x * 0 → 0` 会把 `MUL(0, WHERE(c, x, Invalid))` 连同其有效性一起抹掉；`with_tier2` 把规范化排在项合并之前、ALU 折叠排在比较规则之前，因为每一组都会为下一组暴露出匹配机会（参见[代数化简](./algebraic-simplification.md)）。添加一条规则就意味着要选择它在该顺序中的触发位置。

## 用 Z3 验证重写 {#verifying-rewrites-with-z3}

`schedule/src/z3/`（feature `z3`，可选依赖 `z3 = "0.21"`，需要系统 `libz3`；nix flake 会提供）对重写进行检查，而不是盲目信任：

- `convert.rs` 把 UOp 树翻译为 Z3 项：`CONST`（int、uint、bool；浮点和 `Invalid` 会被拒绝）、作为有界整数的 `DefineVar`、作为满足 `0 <= r < end` 的新变量的 `RANGE`、`Neg`、整数二元算子 `Add, Sub, Mul, FloorDiv, FloorMod, CDiv, CMod, Max, Lt, Eq, Ne`（bool 上的 `And`/`Or`）、整数上的 `WHERE` 和 `MulAcc`，以及作为受 dtype 约束的新变量的 `CAST`（当源的范围能容纳时与其源绑定）。`alu.rs` 赋予 `CDiv`/`CMod` C 语言的截断语义；向下取整除法在其基础上构建。其他一切都是 `ConversionError`。
- `verify_equivalence(original, simplified)` 把两者转换到同一个上下文中并断言 `original != simplified`：`UNSAT` 证明重写正确，`SAT` 返回 `CounterExample::Found { model, .. }`，超时则为 `Unknown`。

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

覆盖范围（`schedule/src/test/`）：经过 `symbolic_simple` 的手写用例（`unit/z3/symbolic_patterns.rs`），经过 `symbolic_simple` 和 `symbolic`、基于 `arb_arithmetic_tree_bounded_up_to` 与 `arb_known_property_graph` 的 proptest 预言测试（`property/oracles.rs`，每项 300–500 个用例），以及结构化符号测试的双重运行 —— 每个用例只要能转换就再用 Z3 复查一次（`unit/symbolic/mod.rs`）。只有 `Found` 会让测试失败；`Unknown` 和 `ConversionFailed` 是可容忍的，另有一个活性测试保证算术核心仍然可以转换。用 `cargo test -p svod-schedule --features z3,proptest` 运行；CI 通过 `nix flake check` 以相同的 feature 运行。

这一证明是在无界整数上、针对有限算子子集的抽样表达式进行的 —— 它是索引化简器的一张强力回归网，而不是对每个模式的验证。
