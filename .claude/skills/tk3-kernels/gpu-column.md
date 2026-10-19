# Measuring a GPU (a `tk3_pr.md` column, a remote or rented box)

Authoritative commands and the line to copy per row: `tk3_pr.md` "How to add a column" (untracked, root of the
integrator's checkout). Hosts, SSH and sync details for the team's boxes live in the user's auto-memory, never in the repo.

## Ground rules

- **`SVOD_DEVICE` on every command**: `CUDA` / `CUDA:0` on NVIDIA, `AMD` / `AMD:N` on Radeon, MI300X or an APU (KFD node
  order: check `cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep gfx_target_version`). Unset with a supported GPU
  attached, every device test fails (`device_target()`); `SVOD_DEVICE=CPU` skips them.
- **One timed process per GPU**, nothing else on it (`nvidia-smi` / `rocm-smi` first). `--test-threads=1` for every probe;
  the agents on one box run their GPU work serially.
- **Measure from a detached worktree** at the exact commit, never from a checkout other agents edit mid-run:
  `git worktree add --detach ~/projects/svod-<name> <sha>`, symlink `data/` and the audio files in, its own `target/`.
- **Fresh store, probes alone, examples before the suite**: `export SVOD_TK3_TUNE_DIR=$PWD/tk3_tune_$(hostname)`; run
  `device_costs_are_physical` alone first (the suite's parallel tests inflate the first probes, +17 % on the trip probes on
  sm_86, and the store keeps the first measurement); time model examples before the parallel suite tunes the store, or
  on a store it has not touched (contended tunes leave worse picks: Nemotron attention 34.6 vs 21.5 µs on MI300X).
- **No `BEAM`** for tk3 rows (tk3 kernels never go through it); a graph-share pass uses the same value on every GPU.
- **Keep the store you produce** and send the device file back (`<arch>_<sms>sm-v<ver>.txt`): `tk3_dump --store`
  reads its probe lines to rank offline.

## Steps

```sh
# 0. host gate (no GPU): every family lowers, keeps its numbers, compiles for each target
SVOD_DEVICE=CPU cargo test -p svod-tk3 --release --lib every_family_lowers_keeps_its_numbers_and_compiles 2>&1 | tee host.txt
# 1. fresh store, probes alone
SVOD_DEVICE=… cargo test -p svod-tk3 --release --lib device_costs_are_physical -- --nocapture 2>&1 | tee costs.txt
# 2. correctness gate: stop and send gate.txt if it fails
SVOD_DEVICE=… cargo test -p svod-tk3 --release --lib families_match_the_interpreter_on_the_device -- --nocapture 2>&1 | tee gate.txt
SVOD_DEVICE=CUDA cargo test -p svod-tk3 --release --lib register_staged_families_match_on_cuda -- --nocapture   # NVIDIA only
# 3. models (Table B) next, while the store holds only uncontended tunes; then the suite
SVOD_DEVICE=… cargo test -p svod-tk3 --release --lib 2>&1 | tee suite.txt; grep -E "FAILED|panicked|test result" suite.txt
# 4. kernel probes (Table A)
for p in gemm_throughput_probe attention_throughput_probe decode_throughput_probe conv_throughput_probe norm_bandwidth_probe; do
  SVOD_DEVICE=… cargo test -p svod-tk3 --release --lib $p -- --ignored --nocapture --test-threads=1 2>&1 | tee probe_$p.txt
done
```

Model rows: each row's command is in `tk3_pr.md` Table B; run twice (the first tunes) and report the second, rows marked
*best of three* three more times. `fallback` where the profile shows graph kernels (`E_…`, `r_…`) in place of `gemm`,
`flash_attention`, `heads`, `layer_norm`/`rms_norm`, `conv`. Send back the logs, the store file, and
`nvidia-smi -q | grep -E "Product|Clocks|Power"` or `rocminfo | grep -E "gfx|Compute Unit|LDS"` once. No numbers for a
device whose gate failed. On an APU, compare only within one round-robin set: it throttles under sustained load and
absolute levels across processes are not comparable.

## Before pushing

The device suite is not enough: `cargo clippy --workspace --all-targets`, `cargo test --workspace --no-run`, and the
`svod-codegen`, `svod-tensor`, `svod-model` lib tests on CPU (a `GemmCfg` field broke `svod-model` tests and an NVPTX
shared alignment broke `svod-codegen` tests while every card's tk3 suite passed).

## On a crash, a hang or wrong numbers

```sh
SVOD_TK3_TUNE=0 SVOD_DUMP_AMD_IR=dump_ir SVOD_DUMP_AMD_ASM=dump_ir SVOD_DUMP_NVPTX_IR=dump_ir TK3_DUMP_LIST=1 \
  RUST_LOG=svod_tk3::plan=debug RUST_BACKTRACE=1 \
  SVOD_DEVICE=… cargo test -p svod-tk3 --release --lib <failing test> -- --nocapture 2> dump_list.txt
```

- The dumps are written on object-cache hits too, one file per IR (`<arch>_<kernel>_<ir digest>.ll`, `.s`).
- A GPU hang: note the last family the gate printed and kill the run.
- AMD `ring gfx_0.0.0 timeout`, then GTT/page faults in every later process: the device is wedged until the box reboots;
  stop timing and tell the user.
- An AMD box's clang needs the AMDGPU target (`clang -print-targets | grep amdgcn`) and clang ≥ 22
  (`runtime/src/amd/compile.rs::MIN_AMD_CLANG_MAJOR`); without it the AMD compile checks print `skipped: …` and pass.

## Rented boxes (vast.ai)

Use Vast's base images (`vastai/base-image:cuda-…`): plain `nvidia/cuda:*` images break SSH (bad ownership of
`authorized_keys`). Containers run as root with no systemd, and `nproc` shows the host's cores. Power caps differ per
rental: note them in the column header. Destroy every instance when its run ends (`vastai destroy instance <id>` asks
for confirmation).
