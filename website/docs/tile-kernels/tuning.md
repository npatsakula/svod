---
sidebar_label: Autotuning
---

# First-Use Autotuning

A kernel's tile table is a search space, not an answer. The GEMM carries three to five tiles
per family, flash attention four per-warp tiles, single-query attention a few K/V splits —
and which one wins depends on the shape, the device's compute-unit count and its clock. So
the first time a device meets a shape, `svod-tk` compiles and times every candidate that
fits, keeps the fastest, and remembers it on disk. `tk/src/tune.rs` is the whole mechanism.

---

## What happens on the first launch

`GemmPolicy::tuned`, `FaPolicy::tuned` and `SqPolicy::tuned` (in `tk/src/kernels/`) run the
same sequence through `TuneStore::select`:

1. **Filter** the table to the candidates that tile the shape (and, for the GEMM, carry the
   requested `Epilogue`). One candidate or none: return the static choice, measure nothing.
2. **Memo.** A process-wide `HashMap<TuneKey, usize>` answers a repeated shape without
   building a kernel — a plan asks the same shape once per node.
3. **Store.** On a memo miss, build every candidate's `SINK` against placeholder buffers and
   fingerprint it (`kernel_fingerprint`). The digests join the store line, so a change to the
   kernel body re-measures. Look the line up in the device's file.
4. **Measure.** On a store miss, `compile_kernel` each candidate on synthetic operands of the
   shape (`Tensor::randn`, cast to the dtype, moved to the device). The first one that built
   lifts the clock — `warm_clock` dispatches it until its time stops falling or 1.5 s has
   passed; a device already under load plateaus in a few runs. Then `round_robin_min` times
   every candidate in turn for three rounds and keeps each one's minimum, so none is judged at
   a clock the others were not.
5. **Keep** the fastest: in the memo, and in the store file. A candidate that cannot build or
   dispatch is skipped; if none can, nothing is cached and the static choice is used.

Only the measured winner reaches the file. `select_with` is the same policy with the
measurement supplied by the caller, which is how the unit tests in `tk/src/test/unit/tune.rs`
exercise it without a GPU.

---

## The key and the store

```rust
// tk/src/tune.rs
pub struct TuneKey {
    pub kernel: &'static str,   // "gemm_nt", "flash_attention", "sq_attention"
    pub device: String,         // "<arch target name>-<compute units>cu"
    pub shape: Vec<usize>,      // the kernel's own shape tuple, dtype width and flags included
    pub config: u64,            // a digest of the candidate set (and anything else the graphs vary with)
}
```

The GEMM keys on `[m, k, n, dtype.bytes(), epilogue.code()]`; flash attention on
`[b, n, h, h_kv, d, causal, mask.code(), dtype.bytes()]`. Changing the table changes
`config`, so a new candidate re-measures.

The store is one file per device and crate version, one line per entry:

```text
<kernel>|<device>|<shape>|<builds digest> <winning index> <ns>
```

at the first of:

| Location | When |
|---|---|
| `$SVOD_TK_TUNE_DIR/` | the variable is set |
| `$XDG_CACHE_HOME/svod/tk_tune/` | else, when `XDG_CACHE_HOME` is set |
| `$HOME/.cache/svod/tk_tune/` | otherwise |

The file name is the device string with non-alphanumerics replaced
(`gfx1201_64cu-v0.1.0.txt`). Writes re-read, merge and rename atomically, so two
processes tuning at once lose at most each other's newest line. An unreadable or unwritable
directory is a miss, never an error; with no writable root the store is memory-only.

---

## Turning it off

| Control | Effect |
|---|---|
| `SVOD_TK_TUNE=0` | no measurement; every policy returns its static choice (`GemmPolicy::cfg`, `FaPolicy::config`, the policy's split) |
| `svod_tk::tune::set_enabled(false)` | the same, from code, overriding the environment for the process — the test harnesses call it so a kernel test does not tune every shape it touches |
| `gemm_nt_with(x, w, cfg)`, `flash_attention_tuned(q, k, v, opts, policy)`, `SqAttentionOpts::split` | bypass the policy for one launch with a chooser of your own |

Tuning is also skipped when the device stamps no dispatch timestamps (`dispatch_gpu_ns` is
`None`), since there would be nothing to compare.

---

## What is tuned

| Kernel | Candidates | Table |
|---|---|---|
| `gemm_nt` | every `GemmCfg` of the family's table that tiles `(m, k, n)` and carries the epilogue | `CUDA_TILES` (2), `RDNA_TILES` (3), `RDNA4_TILES` (5) in `tk/src/kernels/gemm.rs` |
| `flash_attention` | every `(q_blk, kv_blk)` of `FA_TILES` whose K/V double buffers fit shared memory and whose block divides `N` | `[(16,16), (16,32), (16,64), (32,32)]` in `tk/src/kernels/fa.rs` |
| `single_query_attention` | the divisors of `N` nearest the device's resident-wave budget that leave each chunk at least 15 trips | `SqPolicy::candidates` in `tk/src/kernels/sq_attention.rs` |

The static choice each policy falls back to is itself measured, on one part per family:
the GEMM's `GemmPolicy::cfg` prefers the widest tile unless its grid would not fill the
device's compute units `resident` times over; `FaPolicy::tile` picks the taller per-warp tile
only once the launch grid covers the device and the head dim is below the family's bound.
Tuning exists because those crossovers move with the shape.

:::tip[Reading a measurement]
`SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib tune::gemm_first_use -- --ignored` runs the
real sequence against a scratch store and asserts one file, one line, and that the second
request reads it back without measuring. To see what was chosen for a shape in your own run,
read the file: the index is a position in the table named above.
:::

The cost is paid once per shape per device: a few compiles plus roughly two seconds of
timing for a cold GPU. A model whose shapes are bucketed — Qwen3 buckets sequence lengths to
`FLASH_ATTENTION_SEQUENCE_MULTIPLE` — tunes a handful of lines and then runs every batch
from the memo.
