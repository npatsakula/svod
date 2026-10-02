//! FP32 single-query attention for decoder inference.
//!
//! One wave owns one `(batch, head)`. Q stays resident in registers while K/V
//! stream over N; lane `l` owns dimensions `l + j*wave_size`. Dot products use
//! XOR-shuffle all-reduces and a one-pass stable online softmax. Long unmasked
//! attention can split K/V into contiguous chunks and associatively merge their
//! FP32 softmax states. There is no LDS or MFMA.

use std::sync::Arc;

use smallvec::smallvec;
use snafu::ensure;
use svod_dtype::{CudaArch, DType};
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::index::{Idx, flat_index, flat_offset, index_off_gated, load_at};
use crate::scaffold::GlSpec;
use crate::{ArchCaps, ArchSet, Kernel};

/// Architectures on which the scalar shuffle implementation is supported: the CDNA
/// and RDNA parts plus CUDA from Ampere up (the kernel needs only `shfl.sync` and
/// `ex2`).
pub const SQ_ATTENTION_SUPPORTED_ARCHS: ArchSet =
    ArchSet::amd(crate::target::CDNA_RDNA_WMMA).with_cuda_from(CudaArch::from_compute_capability(8, 0));

/// Compile-time masking options for [`single_query_attention`].
#[derive(Clone, Copy, Default)]
pub struct SqAttentionOpts<'a> {
    /// Optional `[B]` i32 valid-key counts. Keys `0..key_lens[b]` are valid.
    ///
    /// Entries must be in `0..=N`: the count is the loop's trip bound and the
    /// renderer compares it unsigned, so a negative one runs the loop four
    /// billion times rather than none.
    pub key_lens: Option<&'a Tensor>,
    /// Also include key `N-1`: the Whisper self-cache layout where the current
    /// token occupies the final slot. Exactly one of this and [`Self::appended`]
    /// is required when `key_lens` is present.
    pub include_last: bool,
    /// The current token's `[B,1,H,D]` key/value, scored after the prefix the
    /// cache holds instead of being read from it.
    ///
    /// A decoder step projects one K/V row per layer and must attend to it
    /// alongside the cache. Splicing it into the cache first copies the whole
    /// cache slice every layer; handing it over separately costs one peeled
    /// online-softmax update. Dtype matches K/V and `H` is the selected head
    /// count, so the appended row is indexed by the *selected* head, not the
    /// packed one. Requires `key_lens`, which then names the prefix alone.
    pub appended: Option<(&'a Tensor, &'a Tensor)>,
    /// Number of contiguous K/V chunks; `None` takes the device's
    /// [`SqPolicy`] choice, measured on first use when tuning is on. Values
    /// above one are supported only for unmasked attention when `N` is
    /// divisible by `split`.
    pub split: Option<usize>,
    /// Optional `[B]` i32 map from a query row to the K/V row it reads.
    ///
    /// Rows that decode the same audio share a cross-attention cache — beam
    /// search runs every hypothesis against one window — but rows belonging to
    /// different windows do not. Pointing each row at its own cache row lets one
    /// copy serve a whole hypothesis set without serializing the windows: the
    /// caller writes the cache once per window instead of once per row, and the
    /// step reads the largest tensor it touches once per window rather than once
    /// per row. Entries must be in `0..K batch`; out-of-range reads are undefined.
    pub cache_map: Option<&'a Tensor>,
}

/// Lanes cooperating on one key's dot product; a wave scores `wave / SUBGROUP`
/// keys per loop trip.
const SUBGROUP: usize = 8;

/// How the unmasked attention is split over K/V on one device.
///
/// The partial kernel is one wave per `(row, head, split)` streaming its chunk
/// of keys, a latency-bound loop; the device needs enough of those waves in
/// flight to reach its bandwidth, and each split costs a merge pass and a
/// launch, so a chunk should not get too short. The budget is the device's own
/// count of resident waves; the floor is a kernel property, in loop trips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SqPolicy {
    pub compute_units: usize,
    /// Waves per compute unit the split aims to keep in flight; `0` keeps one
    /// split, for a caller that wants the kernel unsplit.
    pub waves_per_cu: usize,
    /// Fewest keys a split may hold.
    pub min_chunk: usize,
    /// Keys one wave scores per loop trip; a chunk that is a multiple of it
    /// has no partial tail, which the ranking prefers.
    pub keys_per_trip: usize,
}

/// The splits [`SqPolicy::tuned`] measures on first use: the nearest divisors
/// around the policy's target.
const SQ_CANDIDATES: usize = 4;

/// Fewest loop trips a split may leave a wave. Below it the merge pass and the
/// extra launch outweigh the parallelism: on whisper large-v3's cross
/// attention (b=5, h=20, n=1500, f16 cache) on a wave32 RDNA part every split
/// leaving 60 keys or more sat within 5% of the best, while on a wave32 Ampere
/// part the splits between 2 and 10 did.
const MIN_TRIPS: usize = 15;

/// The `(compute units, waves per CU)` assumed of a device that reports
/// neither. The probe reads the KFD topology or the CUDA limits, both of which
/// can come back empty on a part the launch otherwise supports — a node whose
/// properties omit `simd_per_cu`, sysfs absent in a container — and keeping one
/// split there runs large-v3's 1500-key cross attention as a single
/// latency-bound wave per `(row, head)`. A small discrete part's budget, which
/// puts that geometry near the split the kernel shipped before it read the
/// device.
const UNREPORTED_BUDGET: (usize, usize) = (16, 32);

impl SqPolicy {
    /// A policy for `arch` that aims `compute_units * waves_per_cu` waves at
    /// the device; `waves_per_cu == 0` keeps one split.
    pub fn with_budget(arch: svod_dtype::GpuArch, compute_units: usize, waves_per_cu: usize) -> Self {
        let keys_per_trip = ArchCaps::for_arch(arch).wave_size / SUBGROUP;
        Self { compute_units, waves_per_cu, min_chunk: MIN_TRIPS * keys_per_trip, keys_per_trip }
    }

    /// The policy of the device behind `spec`: its compute units and the waves
    /// each keeps resident ([`crate::target::resident_waves_per_cu`]), or
    /// [`UNREPORTED_BUDGET`] when the backend reports neither.
    pub fn for_device(spec: &svod_dtype::DeviceSpec, arch: svod_dtype::GpuArch) -> Self {
        let budget = crate::target::compute_units(spec).zip(crate::target::resident_waves_per_cu(spec));
        let (compute_units, waves_per_cu) = budget.unwrap_or(UNREPORTED_BUDGET);
        Self::with_budget(arch, compute_units, waves_per_cu)
    }

    /// The divisors of `n` above one that leave every chunk at least
    /// `min_chunk` keys, nearest the wave budget first, a chunk with a partial
    /// tail counting as a little further off. Empty when the policy keeps one
    /// split, or when no split qualifies: the unsplit kernel is the fallback,
    /// never a candidate.
    pub fn candidates(&self, b: usize, h: usize, n: usize) -> Vec<usize> {
        if self.waves_per_cu == 0 || b * h == 0 {
            return Vec::new();
        }
        let target = (self.compute_units * self.waves_per_cu).div_ceil(b * h);
        let distance =
            |s: usize| s.abs_diff(target) + usize::from(!(n / s).is_multiple_of(self.keys_per_trip)) * target / 4;
        let mut splits: Vec<usize> = (2..=n).filter(|s| n.is_multiple_of(*s) && n / s >= self.min_chunk).collect();
        splits.sort_by_key(|&s| distance(s));
        splits.truncate(SQ_CANDIDATES);
        splits
    }

    /// The split for a `[b, h]` query set over `n` keys: the best-ranked
    /// candidate, one when there is none.
    pub fn split(&self, b: usize, h: usize, n: usize) -> usize {
        self.candidates(b, h, n).first().copied().unwrap_or(1)
    }

    /// The split for the geometry as measured on this device ([`crate::tune`]):
    /// every candidate's partial and merge pair is timed together on synthetic
    /// operands and the fastest kept in `store`; [`Self::split`] where fewer
    /// than two candidates exist or nothing measured. The launch entry consults
    /// [`crate::tune::enabled`] before coming here.
    pub(crate) fn tuned(
        &self,
        store: &crate::tune::TuneStore,
        spec: &svod_dtype::DeviceSpec,
        arch: svod_dtype::GpuArch,
        geom: &SqGeom,
        cache_map: bool,
    ) -> usize {
        use std::time::Duration;

        use svod_runtime::benchmark::{CLOCK_WARMUP, round_robin_min, warm_clock};

        let (b, n, h, d) = (geom.b, geom.n, geom.heads.count, geom.d);
        let candidates = self.candidates(b, h, n);
        let fallback = candidates.first().copied().unwrap_or(1);
        if candidates.len() < 2 {
            return fallback;
        }
        // The head offset only moves a constant in the index: every layer of a
        // packed cache measures as one shape.
        let geom = &SqGeom { heads: HeadSelection { offset: 0, ..geom.heads }, ..geom.clone() };
        let caps = ArchCaps::for_arch(arch);
        let block = caps.wave_size as i64;
        let f32 = DType::Float32;
        let placeholder = |shape: &[usize], dtype: &DType| {
            UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, shape.iter().product(), dtype.clone())
        };
        // The store line covers the candidate kernels' graphs, fingerprinted in
        // candidate order; only a memo miss pays for building them.
        let builds = || {
            candidates
                .iter()
                .map(|&splits| {
                    let mut bufs = vec![
                        placeholder(&[b, splits, h, d], &f32),
                        placeholder(&[b, splits, h, 2], &f32),
                        placeholder(&[b, 1, h, d], &f32),
                        placeholder(&[geom.kv_batch, n, geom.heads.total, d], &geom.kv),
                        placeholder(&[geom.kv_batch, n, geom.heads.total, d], &geom.kv),
                    ];
                    if cache_map {
                        bufs.push(placeholder(&[b], &DType::Int32));
                    }
                    let grid = [h as i64, b as i64, splits as i64];
                    let ker = Kernel::new("sq_attention_partial", grid, block, bufs, caps);
                    build_single_query_attention_partial(&ker, geom.clone(), splits, cache_map);
                    crate::kernel_fingerprint(&ker.finish(2)).digest
                })
                .collect()
        };
        let shape = [b, geom.kv_batch, n, h, geom.heads.total, d, geom.kv.bytes(), usize::from(cache_map)];
        let key = crate::tune::TuneKey::new("sq_attention", spec, arch, &shape, &(&candidates, &geom.kv));
        store
            .select_with(&key, candidates.len(), builds, || {
                // The cache keeps its real strides but only the rows the kernel
                // reads — a map of zeros names row 0 — and is filled on the
                // device: timing does not depend on its values.
                let rows = if cache_map { 1 } else { geom.kv_batch };
                let cache = || {
                    Tensor::full(&[rows, n, geom.heads.total, d], ConstValue::Float(0.5), geom.kv.clone())
                        .to(spec.clone())
                };
                let compile = |splits: usize| -> Option<[crate::launch::CompiledLaunch; 2]> {
                    let q = Tensor::randn(&[b, 1, h, d]).ok()?.to(spec.clone());
                    let (k, v) = (cache(), cache());
                    let map = cache_map.then(|| Tensor::zeros(&[b], DType::Int32).to(spec.clone()));
                    let mut ins = vec![&q, &k, &v];
                    ins.extend(map.as_ref());
                    let mut numerator = Tensor::empty(&[b, splits, h, d], f32.clone()).to(spec.clone());
                    let mut stats = Tensor::empty(&[b, splits, h, 2], f32.clone()).to(spec.clone());
                    let geom = geom.clone();
                    let partial = crate::launch::compile_kernel(
                        "sq_attention_partial_tune",
                        [h as i64, b as i64, splits as i64],
                        block,
                        &mut [&mut numerator, &mut stats],
                        &ins,
                        move |ker| {
                            build_single_query_attention_partial(ker, geom, splits, cache_map);
                            ker.finish(2)
                        },
                    )
                    .ok()?;
                    let mut out = Tensor::empty(&[b, 1, h, d], f32.clone()).to(spec.clone());
                    let merge = crate::launch::compile_kernel(
                        "sq_attention_merge_tune",
                        [h as i64, b as i64, 1],
                        block,
                        &mut [&mut out],
                        &[&numerator, &stats],
                        move |ker| {
                            build_single_query_attention_merge(ker, b, h, d, splits);
                            ker.finish(1)
                        },
                    )
                    .ok()?;
                    Some([partial, merge])
                };
                let launches: Vec<Option<[crate::launch::CompiledLaunch; 2]>> =
                    candidates.iter().map(|&splits| compile(splits)).collect();
                // A candidate's time is its partial and merge together.
                let pair_time = |pair: &[crate::launch::CompiledLaunch; 2]| {
                    let mut total = 0;
                    for launch in pair {
                        total += launch.dispatch_gpu_ns().ok().flatten()?;
                    }
                    Some(Duration::from_nanos(total))
                };
                if let Some(first) = launches.iter().flatten().next() {
                    warm_clock(CLOCK_WARMUP, || pair_time(first));
                }
                let time = |i: usize| pair_time(launches[i].as_ref()?);
                round_robin_min(candidates.len(), crate::tune::ROUNDS, None, time)
                    .into_iter()
                    .map(|t| t.map(|t| t.as_nanos() as u64))
                    .collect()
            })
            .map_or(fallback, |i| candidates[i])
    }
}

/// Shape one single-query attention kernel is built for.
///
/// `kv_batch` is `b`, `1` when one K/V cache serves every row, or the number of
/// caches a `cache_map` selects between.
#[derive(Clone)]
pub(crate) struct SqGeom {
    pub(crate) b: usize,
    pub(crate) kv_batch: usize,
    pub(crate) n: usize,
    pub(crate) heads: HeadSelection,
    pub(crate) d: usize,
    /// K/V element type. Scores and the running softmax stay f32; only the
    /// cache narrows, which is what a 1500-key cross cache is mostly made of.
    pub(crate) kv: DType,
}

/// The K/V row a query row reads: its own, row 0 for a single shared cache, or
/// whatever `cache_map` names. The map is bound last so the ABI stays
/// `q, k, v[, key_lens][, cache_map]`.
fn kv_row_index(ker: &Kernel, b: usize, kv_batch: usize, cache_map: bool, batch: &Arc<UOp>) -> Idx {
    if cache_map {
        let map = ker.gl(&[b], DType::Int32);
        return Idx::Uop(load_at(map.uop(), map.shape(), &[Idx::from(batch)]));
    }
    if kv_batch == 1 { Idx::Const(0) } else { Idx::Uop(batch.clone()) }
}

fn cidx(v: i64) -> Arc<UOp> {
    UOp::index_const(v)
}

fn f32c(v: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(v))
}

#[derive(Clone, Copy)]
pub(crate) struct HeadSelection {
    pub(crate) count: usize,
    pub(crate) total: usize,
    pub(crate) offset: usize,
}

/// Which keys a kernel scores, and where the current token's key lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SqMask {
    /// Every key `0..N`; no `key_lens` is bound.
    Whole,
    /// Keys `0..key_lens[b]` plus slot `N-1` of the same cache.
    PrefixAndLast,
    /// Keys `0..key_lens[b]` plus a separately bound `[B,1,H,D]` key/value pair.
    PrefixAndAppended,
}

/// Build the one-wave single-query attention kernel.
///
/// ABI is `out, q, k, v, [key_lens], [appended_k, appended_v], [cache_map]`, with
/// sequence-major `[B,S,H,D]` Q/output and `[B,S,H_total,D]` K/V globals.
/// `kv_batch` is `b`, or `1` when one K/V cache serves every batch row. Beam
/// search decodes each hypothesis against the *same* audio, so the cross
/// attention cache is identical across rows; binding it once and indexing row 0
/// turns a `b`-fold re-read of the largest tensor in the step into a single one.
/// [`SqMask::PrefixAndAppended`] binds the current token's `[B,1,H,D]` K/V
/// separately and peels one online-softmax update for it after the prefix loop.
/// `cache_map` binds the trailing `[b]` i32 global giving the K/V row each query
/// row reads; without it every row reads its own, or row 0 when `kv_batch` is 1.
pub(crate) fn build_single_query_attention(ker: &Kernel, geom: SqGeom, mask: SqMask, cache_map: bool) {
    let SqGeom { b, kv_batch, n, heads, d, kv: kv_dt } = geom;
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    assert!(n > 0, "single-query attention N must be > 0");
    let ept = d / wave;
    let warp = ker.warp();
    let f32 = DType::Float32;

    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[b, 1, heads.count, d], f32.clone())],
        &[
            GlSpec::new(&[b, 1, heads.count, d], f32.clone()),
            GlSpec::new(&[kv_batch, n, heads.total, d], kv_dt.clone()),
            GlSpec::new(&[kv_batch, n, heads.total, d], kv_dt.clone()),
        ],
    );
    let (out, q, k, v) = (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());
    let batch = ker.grid_y();
    let head = ker.grid_x();
    let packed_head = head.add(&cidx(heads.offset as i64));
    let lane = ker.laneid();
    let prefix = (mask != SqMask::Whole).then(|| {
        let lens = ker.gl(&[b], DType::Int32);
        load_at(lens.uop(), lens.shape(), &[Idx::from(&batch)])
    });
    let appended = (mask == SqMask::PrefixAndAppended).then(|| {
        let shape = [b, 1, heads.count, d];
        (ker.gl(&shape, kv_dt.clone()), ker.gl(&shape, kv_dt.clone()))
    });
    let kv_row = kv_row_index(ker, b, kv_batch, cache_map, &batch);

    let q_reg = ker.alloc_reg(ept, f32.clone());
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let scale = f32c(std::f64::consts::LOG2_E / (d as f64).sqrt());

    let mut init = Vec::with_capacity(2 * ept + 2);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let qv = load_at(q.uop(), q.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
            .mul(&scale);
        init.push(flat_index(&q_reg, &[ept], &[Idx::Const(j as i64)]).store(qv));
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let q_reg = q_reg.after(smallvec![initialized.clone()]);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    // A masked launch streams only the valid prefix. Where the current token sits
    // in the cache's final slot the loop runs one trip longer and maps it there;
    // where it arrives in its own global the prefix is the whole loop, which is
    // empty for a row that has decoded nothing yet.
    let lp = match (&prefix, mask) {
        (Some(prefix), SqMask::PrefixAndLast) => ker.loop_dynamic(prefix.add(&cidx(1))),
        (Some(prefix), _) => ker.loop_dynamic(prefix.clone()),
        (None, _) => ker.loop_static(n as i64),
    };
    let loop_index = lp.index().clone();
    let key = match (&prefix, mask) {
        (Some(prefix), SqMask::PrefixAndLast) => {
            UOp::try_where(loop_index.lt(prefix), loop_index.clone(), cidx(n as i64 - 1)).expect("select appended key")
        }
        _ => loop_index,
    };
    let q_loop = q_reg.after(smallvec![key.clone()]);
    let o_loop = o_reg.after(smallvec![key.clone()]);
    let max_loop = max_reg.after(smallvec![key.clone()]);
    let norm_loop = norm_reg.after(smallvec![key.clone()]);

    let mut dot = f32c(0.0);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let qv = load_at(&q_loop, &[ept], &[Idx::Const(j as i64)]);
        let kv =
            load_at(k.uop(), k.shape(), &[kv_row.clone(), Idx::from(&key), Idx::from(&packed_head), Idx::from(dim)])
                .cast(f32.clone());
        dot = dot.add(&qv.mul(&kv));
    }
    let score = warp.wave_reduce_scalar(dot, |a, p| a.add(p));
    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let next_max = old_max.max(&score);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 alpha");
    let beta = score.sub(&next_max).try_exp2().expect("exp2 beta");
    let new_norm = old_norm.mul(&alpha).add(&beta);

    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(next_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)]).store(new_norm);
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let vv =
            load_at(v.uop(), v.shape(), &[kv_row.clone(), Idx::from(&key), Idx::from(&packed_head), Idx::from(dim)])
                .cast(f32.clone());
        let new_o = old_o.mul(&alpha).add(&vv.mul(&beta));
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)]).store(new_o),
        );
    }
    let output_group = UOp::group(output_stores);
    ker.push_store(output_group, o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_max = max_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended.clone()]);
    let streamed = |j: usize| load_at(&final_o, &[ept], &[Idx::Const(j as i64)]);
    let dim_at = |j: usize| lane.add(&cidx((j * wave) as i64));
    let appended_at = |g: &crate::tile::GL, dim: Arc<UOp>| {
        load_at(g.uop(), g.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
            .cast(f32.clone())
    };
    // One peeled online-softmax update folds the current token's key into the
    // state the prefix loop left. An empty prefix leaves `max` at -infinity, so
    // `alpha` is zero and the zeroed accumulator drops out: the appended key
    // alone normalizes to one.
    let (values, denom): (Vec<Arc<UOp>>, Arc<UOp>) = match &appended {
        None => ((0..ept).map(streamed).collect(), load_at(&final_norm, &[1], &[Idx::Const(0)])),
        Some((append_k, append_v)) => {
            let q_final = q_reg.after(smallvec![ended]);
            let mut dot = f32c(0.0);
            for j in 0..ept {
                let qv = load_at(&q_final, &[ept], &[Idx::Const(j as i64)]);
                dot = dot.add(&qv.mul(&appended_at(append_k, dim_at(j))));
            }
            let score = warp.wave_reduce_scalar(dot, |a, p| a.add(p));
            let old_max = load_at(&final_max, &[1], &[Idx::Const(0)]);
            let next_max = old_max.max(&score);
            let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 appended alpha");
            let beta = score.sub(&next_max).try_exp2().expect("exp2 appended beta");
            let denom = load_at(&final_norm, &[1], &[Idx::Const(0)]).mul(&alpha).add(&beta);
            let fold = |j: usize| streamed(j).mul(&alpha).add(&appended_at(append_v, dim_at(j)).mul(&beta));
            ((0..ept).map(fold).collect(), denom)
        }
    };
    let stores: Vec<Arc<UOp>> = values
        .into_iter()
        .enumerate()
        .map(|(j, value)| {
            let dim = dim_at(j);
            flat_index(out.uop(), out.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
                .store(value.try_div(&denom).expect("normalize"))
        })
        .collect();
    ker.push_store(UOp::group(stores), out.uop().clone());
}

/// Build one unnormalized online-softmax state per contiguous K/V split.
///
/// ABI is `numerator, stats, q, k, v`; K/V use `[B,N,H_total,D]`, while
/// outputs are `[B,S,H,D]` and `[B,S,H,2]`, where the final axis of stats is
/// `(max, norm)`.
/// `kv_batch` is `b`, or `1` when one K/V cache serves every batch row. Beam
/// search decodes each hypothesis against the *same* audio, so the cross
/// attention cache is identical across rows; binding it once and indexing row 0
/// turns a `b`-fold re-read of the largest tensor in the step into a single one.
/// `cache_map` binds a trailing `[b]` i32 global (after `key_lens`, when present)
/// giving the K/V row each query row reads; without it every row reads its own,
/// or row 0 when `kv_batch` is 1.
pub(crate) fn build_single_query_attention_partial(ker: &Kernel, geom: SqGeom, splits: usize, cache_map: bool) {
    let SqGeom { b, kv_batch, n, heads, d, kv: kv_dt } = geom;
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    Kernel::assert_divisible(d, SUBGROUP, "split single-query attention D");
    assert!(splits > 1 && n.is_multiple_of(splits), "split attention requires equal non-empty chunks");
    let ept = d / wave;
    let dot_ept = d / SUBGROUP;
    let chunk = n / splits;
    let groups = wave / SUBGROUP;
    let tiles = chunk.div_ceil(groups);
    let warp = ker.warp();
    let f32 = DType::Float32;

    let (outs, ins) = ker.bind_abi(
        &[
            GlSpec::new(&[b, splits, heads.count, d], f32.clone()),
            GlSpec::new(&[b, splits, heads.count, 2], f32.clone()),
        ],
        &[
            GlSpec::new(&[b, 1, heads.count, d], f32.clone()),
            GlSpec::new(&[kv_batch, n, heads.total, d], kv_dt.clone()),
            GlSpec::new(&[kv_batch, n, heads.total, d], kv_dt.clone()),
        ],
    );
    let (numerator, stats) = (outs[0].clone(), outs[1].clone());
    let (q, k, v) = (ins[0].clone(), ins[1].clone(), ins[2].clone());
    let head = ker.grid_x();
    let packed_head = head.add(&cidx(heads.offset as i64));
    let batch = ker.grid_y();
    let kv_row = kv_row_index(ker, b, kv_batch, cache_map, &batch);
    let split = ker.grid_z();
    let lane = ker.laneid();

    let q_reg = ker.alloc_reg(dot_ept, f32.clone());
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let scale = f32c(std::f64::consts::LOG2_E / (d as f64).sqrt());
    let subgroup_lane = warp.subgroup_laneid(SUBGROUP);
    let group = lane.floor_div(&cidx(SUBGROUP as i64));
    let mut init = Vec::with_capacity(dot_ept + ept + 2);
    for j in 0..dot_ept {
        let dim = subgroup_lane.mul(&cidx(dot_ept as i64)).add(&cidx(j as i64)); // lane-contiguous: one 16-byte load per lane
        let qv = load_at(q.uop(), q.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
            .mul(&scale);
        init.push(flat_index(&q_reg, &[dot_ept], &[Idx::Const(j as i64)]).store(qv));
    }
    for j in 0..ept {
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let q_reg = q_reg.after(smallvec![initialized.clone()]);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    let lp = ker.loop_static(tiles as i64);
    let tile_offset = lp.index().mul(&cidx(groups as i64));
    let group_offset = tile_offset.add(&group);
    let valid = group_offset.lt(&cidx(chunk as i64));
    // The tail key is clamped, not gated: a gated load renders as an exec-masked
    // branch whose value is waited on immediately (`s_waitcnt vmcnt(0)` per load),
    // which serializes every iteration of the loop — not just the tail's. Clamping
    // keeps the loads unconditional; the tail's score is still masked to -inf, so
    // its `beta` is zero and the row it re-read contributes nothing.
    let safe_offset = UOp::try_where(valid.clone(), group_offset.clone(), cidx(chunk as i64 - 1)).expect("clamp key");
    let key = split.mul(&cidx(chunk as i64)).add(&safe_offset);
    let q_loop = q_reg.after(smallvec![key.clone()]);
    let o_loop = o_reg.after(smallvec![key.clone()]);
    let max_loop = max_reg.after(smallvec![key.clone()]);
    let norm_loop = norm_reg.after(smallvec![key.clone()]);
    let mut dot = f32c(0.0);
    for j in 0..dot_ept {
        let dim = subgroup_lane.mul(&cidx(dot_ept as i64)).add(&cidx(j as i64)); // lane-contiguous: one 16-byte load per lane
        let qv = load_at(&q_loop, &[dot_ept], &[Idx::Const(j as i64)]);
        let kv =
            load_at(k.uop(), k.shape(), &[kv_row.clone(), Idx::from(&key), Idx::from(&packed_head), Idx::from(dim)])
                .cast(f32.clone());
        dot = dot.add(&qv.mul(&kv));
    }
    let score = warp.subgroup_reduce_scalar(dot, SUBGROUP, |a, p| a.add(p));
    let score = UOp::try_where(valid.clone(), score, f32c(f64::NEG_INFINITY)).expect("mask tail score");
    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let tile_max = warp.wave_reduce_scalar(score.clone(), |a, p| a.max(p));
    let next_max = old_max.max(&tile_max);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 alpha");
    let beta = score.sub(&next_max).try_exp2().expect("exp2 beta");
    let representative = subgroup_lane.eq(&cidx(0));
    let norm_term = UOp::try_where(representative, beta.clone(), f32c(0.0)).expect("one beta per subgroup");
    let tile_norm = warp.wave_reduce_scalar(norm_term, |a, p| a.add(p));
    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(next_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)])
        .store(old_norm.mul(&alpha).add(&tile_norm));
    let group_betas: Vec<_> = (0..groups).map(|g| warp.broadcast_scalar(&beta, (g * SUBGROUP) as i64)).collect();
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.mul(&cidx(ept as i64)).add(&cidx(j as i64)); // lane-contiguous: one 16-byte load per lane
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let mut tile_o = f32c(0.0);
        for (g, group_beta) in group_betas.iter().enumerate() {
            let group_key_offset = tile_offset.add(&cidx(g as i64));
            let group_valid = group_key_offset.lt(&cidx(chunk as i64));
            let safe = UOp::try_where(group_valid, group_key_offset, cidx(chunk as i64 - 1)).expect("clamp v key");
            let group_key = split.mul(&cidx(chunk as i64)).add(&safe);
            let vv = load_at(
                v.uop(),
                v.shape(),
                &[kv_row.clone(), Idx::from(&group_key), Idx::from(&packed_head), Idx::from(dim.clone())],
            )
            .cast(f32.clone());
            tile_o = tile_o.add(&vv.mul(group_beta));
        }
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)])
                .store(old_o.mul(&alpha).add(&tile_o)),
        );
    }
    ker.push_store(UOp::group(output_stores), o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_max = max_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended]);
    let mut numerator_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.mul(&cidx(ept as i64)).add(&cidx(j as i64)); // lane-contiguous: one 16-byte load per lane
        numerator_stores.push(
            flat_index(
                numerator.uop(),
                numerator.shape(),
                &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::from(dim)],
            )
            .store(load_at(&final_o, &[ept], &[Idx::Const(j as i64)])),
        );
    }
    ker.push_store(UOp::group(numerator_stores), numerator.uop().clone());

    let lane_zero = lane.eq(&cidx(0));
    let max_off = flat_offset(stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(0)]);
    let norm_off = flat_offset(stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(1)]);
    let stats_stores = UOp::group(vec![
        index_off_gated(stats.uop(), max_off, lane_zero.clone()).store(load_at(&final_max, &[1], &[Idx::Const(0)])),
        index_off_gated(stats.uop(), norm_off, lane_zero).store(load_at(&final_norm, &[1], &[Idx::Const(0)])),
    ]);
    ker.push_store(stats_stores, stats.uop().clone());
}

/// Merge split online-softmax states without rereading K/V.
pub(crate) fn build_single_query_attention_merge(ker: &Kernel, b: usize, h: usize, d: usize, splits: usize) {
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    let ept = d / wave;
    let f32 = DType::Float32;
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[b, 1, h, d], f32.clone())],
        &[GlSpec::new(&[b, splits, h, d], f32.clone()), GlSpec::new(&[b, splits, h, 2], f32.clone())],
    );
    let (out, numerator, stats) = (outs[0].clone(), ins[0].clone(), ins[1].clone());
    let head = ker.grid_x();
    let batch = ker.grid_y();
    let lane = ker.laneid();
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let mut init = Vec::with_capacity(ept + 2);
    for j in 0..ept {
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    let lp = ker.loop_static(splits as i64);
    let split = lp.index().clone();
    let o_loop = o_reg.after(smallvec![split.clone()]);
    let max_loop = max_reg.after(smallvec![split.clone()]);
    let norm_loop = norm_reg.after(smallvec![split.clone()]);
    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let partial_max =
        load_at(stats.uop(), stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(0)]);
    let partial_norm =
        load_at(stats.uop(), stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(1)]);
    let next_max = old_max.max(&partial_max);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 merge alpha");
    let beta = partial_max.sub(&next_max).try_exp2().expect("exp2 merge beta");
    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(next_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)])
        .store(old_norm.mul(&alpha).add(&partial_norm.mul(&beta)));
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let partial_o = load_at(
            numerator.uop(),
            numerator.shape(),
            &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::from(dim)],
        );
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)])
                .store(old_o.mul(&alpha).add(&partial_o.mul(&beta))),
        );
    }
    ker.push_store(UOp::group(output_stores), o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended]);
    let denom = load_at(&final_norm, &[1], &[Idx::Const(0)]);
    let mut stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let value = load_at(&final_o, &[ept], &[Idx::Const(j as i64)]).try_div(&denom).expect("normalize merge");
        stores.push(
            flat_index(out.uop(), out.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
                .store(value),
        );
    }
    ker.push_store(UOp::group(stores), out.uop().clone());
}

/// Graph-native FP32 single-query attention.
///
/// Q is `[B,1,H,D]`, K/V are `[B,N,H_total,D]`, and output is `[B,1,H,D]`.
/// The first `H` K/V heads are selected.
/// Returns `Ok(None)` when the target is outside [`SQ_ATTENTION_SUPPORTED_ARCHS`].
/// No generic SDPA fallback is performed here.
pub fn single_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: SqAttentionOpts<'_>,
) -> crate::LaunchResult<Option<Tensor>> {
    single_query_attention_packed(q, k, v, 0, opts)
}

/// Graph-native FP32 single-query attention over selected heads in packed K/V.
///
/// Q is `[B,1,H,D]`, K/V are `[B,N,H_total,D]` — or `[1,N,H_total,D]` to serve
/// every batch row from one cache — and heads
/// `head_offset..head_offset+H` are selected without materializing a slice.
pub fn single_query_attention_packed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_offset: usize,
    opts: SqAttentionOpts<'_>,
) -> crate::LaunchResult<Option<Tensor>> {
    let qd = crate::launch::concrete_dims(q, "single-query attention", "q", 4)?;
    let kd = crate::launch::concrete_dims(k, "single-query attention", "k", 4)?;
    let vd = crate::launch::concrete_dims(v, "single-query attention", "v", 4)?;
    let (b, n, h, h_total, d) = (qd[0], kd[1], qd[2], kd[2], qd[3]);
    let dtype = q.uop().dtype();
    let masked = opts.key_lens.is_some();
    let has_map = opts.cache_map.is_some();
    let mask = match (masked, opts.appended.is_some()) {
        (false, _) => SqMask::Whole,
        (true, false) => SqMask::PrefixAndLast,
        (true, true) => SqMask::PrefixAndAppended,
    };
    let kv_dtype = k.uop().dtype();
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };

    ensure!(
        qd[1] == 1,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "Q sequence",
            value: qd[1],
            multiple: 1usize
        }
    );
    // One K/V cache may serve every batch row: beam search decodes each hypothesis
    // against the same audio, so a cross-attention cache is identical across rows.
    let kv_batch = kd[0];
    // With a map, any cache count is addressable: the map names the row. Without
    // one, a cache is either per-row or the single one every row shares.
    ensure!(
        opts.cache_map.is_some() || kv_batch == b || kv_batch == 1,
        crate::launch::OperandDimMismatchSnafu {
            kernel: "single-query attention",
            dim: "K batch (B or 1 without cache_map)",
            a: kd[0],
            b
        }
    );
    ensure!(
        kv_batch > 0,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "K batch (> 0)",
            value: kv_batch,
            multiple: 1usize
        }
    );
    ensure!(
        head_offset <= h_total && h <= h_total - head_offset,
        crate::launch::OperandDimMismatchSnafu {
            kernel: "single-query attention",
            dim: "selected K heads (offset + H <= H_total)",
            a: head_offset.saturating_add(h),
            b: h_total
        }
    );
    ensure!(
        kd[3] == d,
        crate::launch::OperandDimMismatchSnafu {
            kernel: "single-query attention",
            dim: "K head dim D",
            a: kd[3],
            b: d
        }
    );
    for (dim, a, expected) in [
        ("V batch (must match K)", vd[0], kv_batch),
        ("V sequence N", vd[1], n),
        ("V total heads H_total", vd[2], h_total),
        ("V head dim D", vd[3], d),
    ] {
        ensure!(
            a == expected,
            crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim, a, b: expected }
        );
    }
    ensure!(
        n > 0,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "N (> 0)",
            value: n,
            multiple: 1usize
        }
    );
    // The current token's key is either the cache's final slot or its own pair of
    // globals; a masked launch must name exactly one, and an unmasked one neither.
    ensure!(
        masked == (opts.include_last || opts.appended.is_some()) && !(opts.include_last && opts.appended.is_some()),
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "key_lens with exactly one of include_last and appended",
            value: usize::from(opts.include_last) + usize::from(opts.appended.is_some()),
            multiple: 1usize
        }
    );
    if let Some((append_k, append_v)) = opts.appended {
        for (operand, t) in [("appended k", append_k), ("appended v", append_v)] {
            let ad = crate::launch::concrete_dims(t, "single-query attention", operand, 4)?;
            ensure!(
                ad == [b, 1, h, d],
                crate::launch::OperandShapeSnafu {
                    kernel: "single-query attention",
                    operand,
                    expected: vec![b, 1, h, d],
                    got: ad
                }
            );
            ensure!(
                t.uop().dtype() == kv_dtype,
                crate::launch::DtypeSnafu {
                    kernel: "single-query attention",
                    got: t.uop().dtype(),
                    expected: "the K/V cache dtype"
                }
            );
        }
    }
    if let Some(splits) = opts.split {
        ensure!(
            splits > 0 && (!masked || splits == 1) && (masked || n.is_multiple_of(splits)),
            crate::launch::DimMultipleSnafu {
                kernel: "single-query attention",
                dim: "split (unmasked divisor of N; masked requires 1)",
                value: splits,
                multiple: 1usize
            }
        );
    }
    if let Some(map) = opts.cache_map {
        let md = crate::launch::concrete_dims(map, "single-query attention", "cache_map", 1)?;
        ensure!(
            md == [b],
            crate::launch::OperandDimMismatchSnafu {
                kernel: "single-query attention",
                dim: "cache_map B",
                a: md[0],
                b
            }
        );
        ensure!(
            map.uop().dtype() == DType::Int32,
            crate::launch::DtypeSnafu { kernel: "single-query attention", got: map.uop().dtype(), expected: "i32" }
        );
    }
    if let Some(lens) = opts.key_lens {
        let ld = crate::launch::concrete_dims(lens, "single-query attention", "key_lens", 1)?;
        ensure!(
            ld == [b],
            crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim: "key_lens B", a: ld[0], b }
        );
        ensure!(
            lens.uop().dtype() == DType::Int32,
            crate::launch::DtypeSnafu { kernel: "single-query attention", got: lens.uop().dtype(), expected: "i32" }
        );
    }

    crate::launch_custom(
        &q.device(),
        SQ_ATTENTION_SUPPORTED_ARCHS,
        move |_arch| {
            ensure!(
                dtype == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: dtype.clone(), expected: "f32" }
            );
            // The cache may be stored narrower than the query -- it is 1500 keys
            // against one query row, so it owns the traffic -- but K and V must
            // agree, and the softmax still runs in f32.
            let kv_ok = |dt: &DType| *dt == DType::Float32 || *dt == DType::Float16 || *dt == DType::BFloat16;
            ensure!(
                kv_ok(&k.uop().dtype()) && k.uop().dtype() == v.uop().dtype(),
                crate::launch::DtypeSnafu {
                    kernel: "single-query attention",
                    got: k.uop().dtype(),
                    expected: "matching f32, f16 or bf16 K/V"
                }
            );
            Ok(())
        },
        // The kernel loads `d / wave` elements per lane, so a head dim the
        // arch's wave size does not divide is a fit failure of THIS runtime
        // instance (wave is 32 or 64 by arch), not a caller bug: decline to
        // `Ok(None)` and let the caller's generic attention path take over.
        move |arch| d.is_multiple_of(ArchCaps::for_arch(arch).wave_size),
        move |arch| {
            let caps = ArchCaps::for_arch(arch);
            let geom = SqGeom { b, kv_batch, n, heads, d, kv: kv_dtype.clone() };
            // Masked attention has no split; otherwise the caller's, else the
            // device policy's, measured when tuning is on.
            let splits = match opts.split {
                Some(splits) => splits,
                None if masked => 1,
                None => {
                    let spec = q.device();
                    let policy = SqPolicy::for_device(&spec, arch);
                    if crate::tune::enabled() {
                        policy.tuned(crate::tune::TuneStore::global(), &spec, arch, &geom, has_map)
                    } else {
                        policy.split(b, h, n)
                    }
                }
            };
            if splits == 1 {
                let out = Tensor::empty(&[b, 1, h, d], DType::Float32);
                let mut inputs = vec![q, k, v];
                if let Some(lens) = opts.key_lens {
                    inputs.push(lens);
                }
                if let Some((append_k, append_v)) = opts.appended {
                    inputs.extend([append_k, append_v]);
                }
                if let Some(map) = opts.cache_map {
                    inputs.push(map);
                }
                crate::graph_launch(
                    "sq_attention",
                    [h as i64, b as i64, 1],
                    caps.wave_size as i64,
                    out,
                    &inputs,
                    caps,
                    move |ker| {
                        build_single_query_attention(ker, geom.clone(), mask, has_map);
                        ker.finish(1)
                    },
                )
            } else {
                let mut partial_inputs = vec![q, k, v];
                if let Some(map) = opts.cache_map {
                    partial_inputs.push(map);
                }
                let partials = crate::graph_launch_multi(
                    "sq_attention_partial",
                    [h as i64, b as i64, splits as i64],
                    caps.wave_size as i64,
                    vec![
                        Tensor::empty(&[b, splits, h, d], DType::Float32),
                        Tensor::empty(&[b, splits, h, 2], DType::Float32),
                    ],
                    &partial_inputs,
                    caps,
                    move |ker| {
                        build_single_query_attention_partial(ker, geom.clone(), splits, has_map);
                        ker.finish(2)
                    },
                )?;
                crate::graph_launch(
                    "sq_attention_merge",
                    [h as i64, b as i64, 1],
                    caps.wave_size as i64,
                    Tensor::empty(&[b, 1, h, d], DType::Float32),
                    &[&partials[0], &partials[1]],
                    caps,
                    move |ker| {
                        build_single_query_attention_merge(ker, b, h, d, splits);
                        ker.finish(1)
                    },
                )
            }
        },
    )
}
