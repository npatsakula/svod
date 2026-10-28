# A new tk kernel, end to end

The norms are the smallest complete path: kernel `kernels/rows.rs`, gate `ops/shape.rs::norm`, candidates
`ops/config.rs::norm_candidates`, entry `ops/norm.rs`, tests `test/unit/rows.rs` and `test/unit/ops.rs`. Copy their
shape; the excerpts below are verbatim (trimmed) at 2e9d5edb.

## 1. The kernel (`tk/src/kernels/x.rs`, `pub mod x;` in `kernels/mod.rs`)

- `XCfg` (`Clone + Copy + Debug + PartialEq + Eq + Hash`): the knobs the tuner picks. The program states its warp
  grid (`k.warps(br as u32)`, a column; `k.warps(WarpGrid { rows, cols })`) and each ring
  (`k.pipeline(extent, Ring { stages, unroll, crews }, …)`, or a slot count); the launch takes
  `Lowering::new(&device)`, or the config's `lowering` where it picks the core (`GemmCfg`, `FaCfg`: `Core::Group` →
  the warpgroup atom). A mainloop kernel reuses `gemm::mainloop` with `GemmCfg::ring()`.
- `XSpec`: shapes, `batch: Batch`, options, `cfg: XCfg`. Size every param at `batch.capacity()`.
- `pub fn x<T: Elem>(spec: &XSpec) -> Program`. The `Kernel::new` name is what profiles, `assert_kernel`, the IR dump
  files and `.kd` descriptors show. Param order is the launch order (inputs, then outputs).
- Ragged edges are view bounds (`bound(len, tile)` from `kernels/mod.rs`); only a reduction dim must divide its step.

```rust
// kernels/rows.rs: bounds instead of padding, math in f32, one rounding at the store
let x = k.param::<T>("x", ParamKind::In, elems);
let (gz, bb) = batch.axis(&mut k);                     // Batch::Static(n) | a variable → grid z
k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
k.warps(br as u32);
let row0 = k.block(0) * br;
let at = |k: &mut Kernel, p: ParamRef<T>| {
    let v = k.view(p, batch_offset(&bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
    k.at(v, row0.clone(), 0)                            // rows past `rows` read 0, their stores drop
};
let xv = at(&mut k, x);
let mut xs = load_f32(&mut k, xv);
// … reduce(.., Axis::Row, ReduceOp::Sum), unary(.., UnaryOp::Rsqrt), binary(..) …
let y = k.cast::<F32, T>(y);
let ov = at(&mut k, out);
k.store(ov, y);
k.finish()
```

## 2. The gate (`ops/shape.rs`)

```rust
pub fn norm(planner: Option<&Planner>, dtypes: &[DType], x: Option<&Extent>) -> Plan<NormCfg> {
    plan(|| {
        gate(planner, dtypes)?;                  // Fallback::Target / Dtype (16-bit, one type, a matrix core)
        let x = x.ok_or(Fallback::Symbolic)?;    // a symbolic dim other than a bound leading one
        let d = reduced(x)?;
        if x.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        candidates(config::norm_candidates(d).into(), Fallback::Shape)
    })
}
```

Host-test it with a `test_case` table in `test/unit/ops_plan.rs` (`linear_plans` and `norm_plans` are the models:
`want: Result<(), Fallback>` per shape, dtype and target: `sm86()`, `targets::amd(AmdArch::Gfx1201, 64)`, …).

## 3. Candidates (`ops/config.rs`)

A fixed list when the cost model has nothing to rank:

```rust
pub fn norm_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (256..=2048).contains(&d)) {
        return vec![];
    }
    ROW_WARPS.map(|br| NormCfg { br }).to_vec()      // search bound: [4, 8, 16]
}
```

A ranked lattice when it does: a key like `Problem`/`Attention` with
`admit(&self, &Planner, &cfg) -> Result<(f64, u32), Rejected>` (ns of SM time from `Costs`, layout registers; the
`Rejected` reason otherwise) and a ranking cached per planner like `Planner::gemm_ranked` (`papaya::HashMap`), sorted
through `sorted(..)` so it is logged under `svod_tk::plan`. Every config also needs `impl Tiled for XCfg`
(`tile()` = the config minus its pipeline knobs, `twins(ranked)` = the ranked configs sharing that tile,
`represents()` false for variants only the twins stage times). Name every new constant with its kind (model, search
bound, measurement, numeric; `lib.rs` doc). Pin the ranking on the host in `ops_plan.rs` under declared costs
(`costs()`, `gfx1151_costs()`): which tiles lead and why `admit` drops the rest. `tk_dump` shows the same ranking for
any card (add an `--op` arm there if the kernel should be inspectable offline).

## 4. The entry (`ops/x.rs`, re-exported from `ops/mod.rs`)

From `ops/norm.rs` (shape validation first; `Err` only where the graph op would also fail):

```rust
let (ext, var) = extent(&xs).unzip();
let planner = super::planner(&x.device());
let dtypes: Vec<_> = [Some(x), residual, Some(w), b].into_iter().flatten().map(Tensor::dtype).collect();
let Some(cfgs) = super::kernel(op, shape::norm(planner.as_deref(), &dtypes, ext.as_ref())) else {
    return graph(kind, x, residual, w, b, eps).context(GraphSnafu { op });   // `graph … why` is logged
};
// …
let cfg = Tuner::new(&planner.device, x.dtype(), &shape, (&batch, kind, residual_, bias), |cfg| {
    vec![(typed!(x.dtype(), kernel, &spec(cfg)), cfg.lowering(&planner.device))]
})
.tuned(op, &cfgs, <[_]>::to_vec);
let out = output(&ext.dims, &var, x.dtype()).context(GraphSnafu { op })?;
// …
let mut outs = launch::graph_launch_all(typed!(x.dtype(), kernel, &spec), &cfg.lowering(&planner.device), &ins)
    .context(LaunchSnafu { op })?;
```

- `Tuner::new(device, dtype, shape, salt, build)`: `shape` and `salt` (whatever else the programs vary with) key the
  store; `build` returns a `tune::Candidate` (`Vec<(Program, Lowering)>`: several programs for a split + merge).
- `tuned(op, &ranked, expand)`: `expand` adds variants (key splits) to the admitted shortlist; `<[_]>::to_vec` adds none.
- `output(..)` allocates at capacity with the live batch in dim 0, device-local (`kernel_buffer`).
- Keep `fn graph(..)` `pub(crate)`: the ops tests compare against it.

## 5. Tests (`tk/src/test/unit/`, the module added to `mod.rs`)

| Test | Template |
|---|---|
| program vs f64 reference, any shape | `rows::norm_interpreter_matches_the_reference` (proptest; `interp::run(&prog, params, &vars)`) |
| lowering keeps the result and compiles, every target | add the family to `targets::families` so `every_family_lowers_keeps_its_numbers_and_compiles`, `every_candidate_lowers` and the device gate `families_match_the_interpreter_on_the_device` cover it; target facts go in its `Facts` |
| kernel vs interpreter on device | `rows::norms_match_on_device` (`let Some(target) = super::device_target() else { return };`) |
| every admitted config on device | `device::every_gemm_candidate_matches_the_interpreter` via `super::tuned_on_device` |
| op vs graph fallback | `ops::linear_matches_the_graph`, `ops::norms_match_the_graph` (`untuned(f)` = `tune::scoped(false, f)` behind the device gate, `assert_kernel(&y, "<name>")`) |
| batch variable | `ops::norm_under_a_batch_variable` |
| no barrier added to the shipped kernel | `sync::shipped_kernels_need_no_barriers` (add a `test_case`) |
| perf probe | `#[ignore = "perf probe: needs a GPU"]`, print, never assert (`rows::norm_bandwidth_probe`) |

Host IR of the new kernel for any target, no GPU: `crate::lower::render(prog, &lowering)`; compile it for AMD with
`svod_runtime::amd::compile_ir_to_amd_object(&ir, arch)` / to PTX with `svod_runtime::cuda::compile_ir_to_ptx(&ir, arch)`,
as the sweep does.

Then the whole-workspace check and the perf protocol in SKILL.md: model crates build tk configs, codegen renders their
kernels.
