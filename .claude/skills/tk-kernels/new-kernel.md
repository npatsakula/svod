# A new tk kernel, end to end

The norms are the smallest complete path: kernel `kernels/rows.rs`, gate `ops/shape.rs::norm`, candidates
`ops/config/norm.rs::norm_candidates`, entry `ops/norm.rs`, tests `test/unit/rows.rs` and `test/unit/ops.rs`. Copy their
shape; the excerpts below are trimmed from them. The norms rank nothing, so `NormSpec` carries its config; a ranked
family's spec is admitted instead (below).

## 1. The kernel (`tk/src/kernels/x.rs`, `pub mod x;` in `kernels/mod.rs`)

- `XCfg` (`Clone + Copy + Debug + PartialEq + Eq + Hash`): the knobs the tuner picks. The program states its warp
  grid (`k.warps(br as u32)`, a column; `k.warps(WarpGrid { rows, cols })`, before any `lone_warps`) and each ring
  (`let ring = k.ring::<T>(name, slot, Schedule::ring(stages, unroll))`, or a slot count: `schedule.stages()` slots;
  `ring_runs` for one run of them per warp), and a pipeline runs as the ring it is handed says
  (`k.pipeline(extent, ring, init, produce, consume)`; a second ring its stages reach holding fewer slots is
  `lower::Error::SharedPast`); the launch takes
  `Lowering::new(&device)`, or the config's `lowering` where it picks the core (`GemmCfg`, `FaCfg`: `Core::Group` →
  the warpgroup atom). A mainloop kernel reuses `gemm::mainloop` with `GemmCfg::ring()`.
- `XSpec`: shapes, `batch: Batch`, options, no config. A ranked family implements `kernels::Spec` (`type Cfg = XCfg`)
  and the crate's `Admits` (`type Problem` = its `Family`, `fn problem(&self)` derived from the spec alone), plus
  `pub fn admitted(self, planner: &Planner, cfg: XCfg) -> Result<Admitted<'_, Self>, Rejected>` (`self.admit(..)`:
  feasible + claim). `Admitted<'t, XSpec>` owns the spec and its config, read-only (`Deref` to the spec,
  `.cfg()`), so the kernel fn checks none of what admission checked; a config no planner admitted is taken only
  through `unsafe Admitted::assume(spec, cfg)`; tests, `sweep.rs` and `rank.rs` go through `sweep::assumed`. The op
  layer's `tuned`/`probed` take the `XSpec` and return the `Admitted` pick. Size every param at `batch.capacity()`.
- `pub fn x<T: Elem>(spec: &Admitted<XSpec>) -> Program` is `Kernel::record(name, |k| x_body::<T>(k, spec))` with
  `fn x_body<'k, T: Elem>(k: &mut Kernel<'k>, spec: &Admitted<XSpec>)` (`let cfg = spec.cfg();`; a fixed-list
  family like the norms takes `&XSpec` with its `cfg` field): every handle carries the kernel's brand `'k`, so a
  helper that takes or returns one names it (`fn at<'k, T: Elem, A: Access>(k: &mut Kernel<'k>, p: ParamRef<'k, T, A>,
  …) -> Gmem<'k, T, A>`); a closure cannot (`|k: &mut Kernel<'_>|` is higher-ranked and rejects captured handles). The
  name is what profiles, `assert_kernel`, the IR dump files and `.kd` descriptors show (`Program::name()`; a recorded
  program is read-only outside the crate: `params()`, `body()`). Param order is the launch order (inputs, then
  outputs).
- Ragged edges are view bounds (`bound(len, tile)` from `kernels/mod.rs`); only a reduction dim must divide its step.

```rust
// kernels/rows.rs: bounds instead of padding, math in f32, one rounding at the store
let x = k.input::<T>("x", elems);
let (gz, bb) = batch.axis(k);                          // Batch::Static(n) | a variable → grid z
k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
k.warps(br as u32);
let row0 = k.block(0) * br;
fn at<'k, T: Elem, A: Access>(k: &mut Kernel<'k>, p: ParamRef<'k, T, A>, bb: &Option<Sc<'k>>, [rows, br, d]: [usize; 3], row0: &Sc<'k>) -> Gmem<'k, T, A> {
    let v = k.view(p, batch_offset(bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
    k.at(v, row0.clone(), 0)                            // rows past `rows` read 0, their stores drop
}
let xv = at(k, x, &bb, [rows, br, d], &row0);
let mut xs = load_f32(k, xv);
// … reduce(.., Axis::Row, ReduceOp::Sum), unary(.., UnaryOp::Rsqrt), binary(..) …
let y = k.cast::<F32, T>(y);
let ov = at(k, out, &bb, [rows, br, d], &row0);
k.store(ov, y);
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

## 3. Candidates (`ops/config/`)

A fixed list when the cost model has nothing to rank:

```rust
pub fn norm_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_multiple_of(256) && (256..=2048).contains(&d)) {
        return vec![];
    }
    ROW_WARPS.map(|br| NormCfg { br }).to_vec()      // search bound: [4, 8, 16]
}
```

A ranked lattice when it does: a problem like `GemmProblem`/`AttentionProblem` (a `Family`) with
`admit(&self, &Planner, &cfg) -> Result<(f64, u32), Rejected>` (ns of SM time from `Costs`, layout registers; the
`Rejected` reason otherwise) and a ranking cached per planner like `Planner::gemm_ranked` (`papaya::HashMap`), sorted
through `sorted(..)` so it is logged under `svod_tk::plan`. Every config also needs `impl Tiled for XCfg`
(`tile()` = the config minus its pipeline knobs, `twins(ranked)` = the ranked configs sharing that tile,
`represents()` false for variants only the twins stage times). Name every new constant with its kind (model, search
bound, measurement, numeric; `lib.rs` doc). Pin the ranking on the host in `ops_plan.rs` under declared costs
(`costs()`, `sm86_costs()`, `gfx1201_costs()`, `gfx942_costs()`, `gfx1151_costs()`): which tiles lead and why `admit` drops
the rest. `tk_dump` shows the same ranking for any card (add an `--op` arm there if the kernel should be inspectable offline).

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
let spec = NormSpec { norm: kind, rows, d, batch, eps, residual: residual_, bias, cfg: cfgs[0] };
let spec = tuned(TuneStore::global(), &planner, x.dtype(), &cfgs, spec);
let out = output(&ext.dims, &var, x.dtype()).context(GraphSnafu { op })?;
// …
let mut outs = launch::graph_launch_all(typed!(x.dtype(), kernel, &spec), &Lowering::new(&planner.device), &ins)
    .context(LaunchSnafu { op })?;

// `tuned`, public for the tests and the sweep:
pub fn tuned(store: &TuneStore, planner: &Planner, dtype: DType, ranked: &[NormCfg], spec: NormSpec) -> NormSpec {
    let shape = [spec.batch.capacity(), spec.rows, spec.d];
    let salt = (&spec.batch, spec.norm, spec.residual, spec.bias);
    let build = |cfg| vec![(typed!(dtype, kernel, &NormSpec { cfg, ..spec.clone() }), Lowering::new(&planner.device))];
    let cfg = Tuner::new(store, &planner.device, dtype.clone(), &shape, salt, build).tuned(op(spec.norm), ranked, <[_]>::to_vec);
    NormSpec { cfg, ..spec }
}
```

- `Tuner::new(store, device, dtype, shape, salt, build)`: `shape` and `salt` (whatever else the programs vary with) key the
  store; `build` returns a `tune::Candidate` (`Vec<(Program, Lowering)>`: several programs for a split + merge, timed
  with a `LAUNCH` added per launch past the first, `tune::launched`). A ranked family's `build` admits each config
  for its spec (`spec.clone().admit_ranked(planner, cfg)`), and its `tuned` returns `spec.admit_ranked(planner, pick)`.
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
| perf benchmark | a `tk_bench` command (`tk/examples/tk_bench.rs`: `gemm_bench` builds each config's candidate with `ops::candidates`, times it with `tune::measure`), printing, never asserting; an `#[ignore = "perf probe: needs a GPU"]` test only when its rows feed a fit (`device::gemm_tail_probe`) |

Host IR of the new kernel for any target, no GPU: `crate::lower::render(prog, &lowering)`; compile it for AMD with
`svod_runtime::amd::compile_ir_to_amd_object(&ir, arch)` / to PTX with `svod_runtime::cuda::compile_ir_to_ptx(&ir, arch)`,
as the sweep does.

Then the whole-workspace check and the perf protocol in SKILL.md: model crates build tk configs, codegen renders their
kernels.
