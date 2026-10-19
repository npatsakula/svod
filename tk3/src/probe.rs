//! What a device's primitives cost, measured on it at first use and kept in
//! the tune store. Each cost is a pair of programs timed together: a probe
//! that does everything its baseline does plus `units` of the primitive, so
//! the difference is the primitive alone. A throughput probe fills the device
//! and its difference is scaled to one SM's time per unit. A relayout-class
//! probe (lane moves, shared round trips, scattered loads)
//! runs one wave per SIMD on a dependent chain, and `latency` one wave per
//! SM: their differences are what one wave waits per unit.

use snafu::{OptionExt, Snafu};
use svod_dtype::default_device::default_device;
use svod_dtype::{GpuArch, ScalarDType};

use crate::atoms::{Costs, Device, Target};
use crate::build::*;
use crate::ir::*;
use crate::layout::{Dim, Layout};
use crate::layouts::{TileLayout, WarpGrid, natural};
use crate::lower::Lowering;
use crate::tune::{self, Built, Candidate, ScratchPool, TuneKey, TuneStore};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("{arch:?} reports no SM count, so a probe cannot scale its time to one SM"))]
    NoSms { arch: GpuArch },
    #[snafu(display("{probe} could not be timed on the device in {TRIES} tries; the last: {last}"))]
    Unmeasured { probe: &'static str, last: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Measurement: bytes the streaming probes read once per run: larger than
/// any device cache (MI300X's Infinity Cache is 256 MB), so every run streams from DRAM.
pub(crate) const STREAM: usize = 512 << 20;
/// Measurement: trips of a lone wave at most: a few ms of waiting.
const LATENCY_TRIPS: usize = 16 << 10;
/// Measurement: trips of the trip probes: enough that their fixed costs dwarf
/// a launch.
const TRIPS: usize = 4096;
/// Measurement: timings of a pair, compiled once, before a probe gives up: one with a
/// weak signal ([`strong`]) or a lost stamp is timed again.
const TRIES: usize = 3;
/// Measurement: warps of a throughput probe's block.
const WARPS: u32 = 4;
/// Measurement: a throughput probe's warp grid: [`WARPS`] rows of warps.
const ROWS: WarpGrid = WarpGrid { rows: WARPS, cols: 1 };

/// One measurement: `candidates` are `[baseline, probe]`, the probe doing
/// `units` more of the primitive.
#[derive(Clone, Debug)]
pub(crate) struct Probe {
    /// `probe:<field>`, the tune store's op.
    pub(crate) op: &'static str,
    pub(crate) dtype: ScalarDType,
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) units: f64,
    /// What the difference is scaled by before dividing by `units`: the SM
    /// count for a throughput probe, else 1.
    pub(crate) scale: f64,
}

/// Whether a pair's signal (probe less baseline, ns) is at least half the
/// probe's own time: else the probe times its scaffolding more than its
/// primitive, and noise in either decides the value.
pub(crate) fn strong(base_ns: u64, probe_ns: u64) -> bool {
    probe_ns >= 2 * base_ns
}

/// A device's measured costs, and the fields its probes could not measure,
/// left at [`Costs::default`]'s zero: the planner ranks with what it has
/// rather than leaving the device to the graph, and the device gate holds
/// the list empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    pub costs: Costs,
    pub unmeasured: Vec<&'static str>,
}

/// The costs of the default device, whose target is `target`: every field
/// measured once per device, compiler and probe program, then read from the
/// tune store. Only a target without an SM count, which no probe can scale
/// to, is an error.
pub fn costs(target: &Target) -> Result<Measured> {
    let probes = probes(target)?;
    let compiler = tune::compiler(&default_device());
    let mut measured = assemble(target, probes, |probe| self::measured(target, compiler, probe));
    let Measured { costs, unmeasured } = &mut measured;
    // The cache ladder is read against the DRAM rate; without one, or when a
    // footprint fails, the cache is unmeasured and fills are priced as DRAM.
    let ladder = if unmeasured.contains(&"global_byte") { None } else { cache(target, compiler, costs.global_byte)? };
    (costs.l2_byte, costs.l2_bytes) = ladder.unwrap_or_else(|| {
        unmeasured.push("l2_bytes");
        (costs.global_byte, 0.0)
    });
    // A plan prices a trip's fragment reads by `smem_byte`, so a trip is
    // what its pipeline adds besides reading its tile from the ring.
    let fragment = (trip_tile(target).elems() * 4) as f64 * costs.smem_byte;
    costs.trip = (costs.trip - fragment).max(0.0);
    costs.trip_rolled = (costs.trip_rolled - fragment).max(0.0);
    Ok(measured)
}

/// `probes`, in [`Costs::fields`] order, each measured by `measure` into its
/// field; a probe that fails leaves its field at zero, names itself in
/// `unmeasured`, and is warned about once.
pub(crate) fn assemble(target: &Target, probes: Vec<Probe>, mut measure: impl FnMut(Probe) -> Result<f64>) -> Measured {
    let mut costs = Costs::default();
    let fields = costs.fields();
    assert_eq!(fields.len(), probes.len(), "a probe per field");
    let mut unmeasured = vec![];
    for ((name, value), probe) in fields.into_iter().zip(probes) {
        assert_eq!(probe.op.strip_prefix("probe:"), Some(name), "probes in field order");
        match measure(probe) {
            Ok(v) => *value = v,
            Err(e) => {
                tracing::warn!("{}: {e}; {name} costs nothing in the ranking", target.arch.target_name());
                unmeasured.push(name);
            }
        }
    }
    Measured { costs, unmeasured }
}

/// A probe's value in ns per unit, stored in femtoseconds: a shared byte
/// costs a few picoseconds, which whole picoseconds would round by a third.
/// The pair compiles once and is timed until a timing holds.
fn measured(target: &Target, compiler: u64, probe: Probe) -> Result<f64> {
    let Probe { op, dtype, candidates, units, scale } = probe;
    let programs = tune::fingerprint(&candidates);
    let key = TuneKey::new(op, target, compiler, dtype, &[], &programs);
    let mut last = String::from("not run");
    let fs = TuneStore::global().value(
        &key,
        || programs,
        || {
            let mut built = tune::compile(candidates, &ScratchPool::default());
            (0..TRIES).find_map(|_| match timed(&mut built, scale / units) {
                Ok((fs, base_ns, probe_ns)) => {
                    tracing::debug!(op, base_ns, probe_ns, fs, "probe pair timed");
                    Some(fs)
                }
                Err(e) => {
                    last = e;
                    None
                }
            })
        },
    );
    Ok(fs.context(UnmeasuredSnafu { probe: op, last })? as f64 * 1e-6)
}

/// One timing of a compiled pair: its difference times `per`, in
/// femtoseconds, with the pair's own nanoseconds, or why it does not hold.
fn timed(built: &mut [Built], per: f64) -> Result<(u64, u64, u64), String> {
    match tune::time(built)[..] {
        [Some(base_ns), Some(probe_ns)] if strong(base_ns, probe_ns) => {
            Ok((((probe_ns - base_ns) as f64 * per * 1e6).round() as u64, base_ns, probe_ns))
        }
        [Some(base_ns), Some(probe_ns)] => {
            let signal = (probe_ns as f64 - base_ns as f64) / probe_ns as f64;
            Err(format!("a signal of {signal:.3} of the probe's time (baseline {base_ns} ns, probe {probe_ns} ns)"))
        }
        _ => Err("a program failed to build or lost its stamps".into()),
    }
}

/// Measurement: the footprints the cache probes read [`STREAM`] bytes from,
/// smallest first: from the size of a small cache to half the stream, past
/// every device's cache (MI300X's Infinity Cache is 256 MB), so a probe
/// still reads each byte of a footprint twice.
const FOOTPRINTS: [(&str, usize); 9] = [
    ("probe:cached_1m", 1 << 20),
    ("probe:cached_2m", 2 << 20),
    ("probe:cached_4m", 4 << 20),
    ("probe:cached_8m", 8 << 20),
    ("probe:cached_16m", 16 << 20),
    ("probe:cached_32m", 32 << 20),
    ("probe:cached_64m", 64 << 20),
    ("probe:cached_128m", 128 << 20),
    ("probe:cached_256m", 256 << 20),
];

/// The cache in front of DRAM, as [`climbed`] reads the ladder of
/// footprints; `None`, with a warning, when a footprint cannot be measured.
fn cache(target: &Target, compiler: u64, global_byte: f64) -> Result<Option<(f64, f64)>> {
    let ladder = climbed(cache_probes(target)?, global_byte, |probe| measured(target, compiler, probe));
    Ok(ladder.map_err(|e| tracing::warn!("{}: {e}; the cache is unmeasured", target.arch.target_name())).ok())
}

/// The ladder of `probes` (smallest footprint first) measured until it has
/// [`settled`] past the cache, read by [`held`]: a small cache's ladder
/// stops after a few footprints, and no run streams a large one for
/// nothing.
pub(crate) fn climbed(
    probes: Vec<(usize, Probe)>,
    global_byte: f64,
    mut measure: impl FnMut(Probe) -> Result<f64>,
) -> Result<(f64, f64)> {
    let mut rates = Vec::with_capacity(probes.len());
    for (footprint, probe) in probes {
        if settled(&rates, global_byte) {
            break;
        }
        rates.push((footprint, measure(probe)?));
    }
    Ok(held(&rates, global_byte))
}

/// Whether the ladder's last two footprints both read nearer DRAM's rate
/// than the cache's: one footprint past the knee could be noise or a
/// second cache level; two are past the cache ([`held`] counts neither).
pub(crate) fn settled(rates: &[(usize, f64)], global_byte: f64) -> bool {
    rates.len() >= 2 && rates[rates.len() - 2..].iter().all(|(_, r)| *r > mid(rates, global_byte))
}

/// The rate between the cache's and DRAM's on a log scale.
fn mid(rates: &[(usize, f64)], global_byte: f64) -> f64 {
    (rates[0].1.min(global_byte) * global_byte).sqrt()
}

/// What a ladder of `(footprint, ns a byte)` says of the cache in front of
/// DRAM: a streamed byte's cost from the smallest footprint, which stays in
/// it, and its bytes, the most any footprint shows it answering: a
/// footprint read at a rate between the cache's and DRAM's was answered
/// for that share of its bytes (the rates mix linearly), so the cache holds
/// about the share times the footprint; a footprint that fits shows its own
/// size, one far past the cache shows the cache's, and a slower cache level
/// counts for the share it answers. No cache faster than DRAM: a byte costs
/// DRAM's at most, and nothing is held.
pub(crate) fn held(rates: &[(usize, f64)], global_byte: f64) -> (f64, f64) {
    let near = rates[0].1.min(global_byte);
    // A cache no faster than DRAM answers nothing.
    let answered = |r: f64| if global_byte > near { ((global_byte - r) / (global_byte - near)).max(0.0) } else { 0.0 };
    let bytes = rates.iter().map(|&(f, r)| f as f64 * answered(r)).fold(0.0, f64::max);
    (near, bytes)
}

/// The tiles a cache probe's region of `footprint` bytes holds: the largest
/// prime count of [`CACHE_TILE`]s that fits, so blocks any stride apart read
/// distinct tiles whatever order a dispatcher deals them to its SMs (a
/// power-of-two count against 64 CUs dealt in turn put every block of a CU
/// on one tile, read from its own L0), and a sweep of any block count the
/// prime does not divide reaches every tile.
pub(crate) fn region_tiles(footprint: usize) -> usize {
    let most = footprint / (CACHE_TILE.elems() * 4);
    (2..=most).rev().find(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)).expect("a footprint of tiles")
}

/// Measurement: the tile a cache probe's blocks read: 16 KB of i32.
const CACHE_TILE: Shape = Shape::new(32, 128);

/// The cache probes of `target` with their footprints, smallest first.
pub(crate) fn cache_probes(target: &Target) -> Result<Vec<(usize, Probe)>> {
    let p = Probes::new(target)?;
    Ok(FOOTPRINTS.iter().map(|&(op, footprint)| (footprint, p.cached(op, footprint))).collect())
}

/// Every probe of `target`, in [`Costs::fields`] order.
pub(crate) fn probes(target: &Target) -> Result<Vec<Probe>> {
    let p = Probes::new(target)?;
    Ok(vec![
        p.smem_byte(),
        p.global_byte(),
        p.scatter(),
        p.step(),
        p.step_lone(),
        p.score(),
        p.trips("probe:trip", true),
        p.trips("probe:trip_rolled", false),
        p.lane_move(),
        p.smem_trip(),
        p.latency(),
    ])
}

/// The trip probes' i32 tile: a 16-byte chunk per thread, eight to a row.
fn trip_tile(target: &Target) -> Shape {
    Shape::new((WARPS * target.wave) as usize / 8, 32)
}

struct Probes<'a> {
    target: &'a Target,
    /// Blocks of a relayout-class probe, and what a throughput probe's
    /// difference is scaled by.
    sms: u32,
    /// Blocks of [`WARPS`] that fill every SIMD's wave slots four times over.
    blocks: usize,
    /// A relayout-class probe's warps: one per SIMD.
    warps: u32,
}

impl<'a> Probes<'a> {
    fn new(target: &'a Target) -> Result<Self> {
        let sms = target.sms.context(NoSmsSnafu { arch: target.arch })?;
        let o = target.occupancy;
        let blocks = (4 * sms * o.simds * o.waves / WARPS) as usize;
        Ok(Self { target, sms, blocks, warps: o.simds })
    }

    fn lowering(&self, grid: WarpGrid, unroll: bool) -> Lowering {
        Lowering::uniform(&Device { target: self.target.clone(), costs: Costs::default() }, unroll, grid)
    }

    /// A relayout-class probe's lowering: a block per SM of a warp per SIMD,
    /// one wave on each, so a wave waits out every dependent step itself (and
    /// a barrier spans waves, as in a kernel's block), as a relayout between
    /// two products does; its value is that wait per unit of one wave.
    fn chain(&self, costs: Costs) -> Lowering {
        let mut lowering = self.lowering(WarpGrid { rows: self.warps, cols: 1 }, false);
        lowering.device.costs = costs;
        lowering
    }

    /// `[baseline, probe]` from `build(false)` and `build(true)`, lowered
    /// alike; `per_sm` scales the difference to one SM's time.
    fn make(
        &self,
        op: &'static str,
        dtype: ScalarDType,
        units: f64,
        per_sm: bool,
        lowering: Lowering,
        build: impl Fn(bool) -> Program,
    ) -> Probe {
        let candidates = [false, true].map(|on| vec![(build(on), lowering.clone())]).into();
        let scale = if per_sm { self.sms as f64 } else { 1.0 };
        Probe { op, dtype, candidates, units, scale }
    }

    /// Operand bytes read from shared memory: an f32 tile loaded and summed
    /// each iteration, where the baseline sums a register tile in the same
    /// loop (f32, so the compiler cannot fold its sums into a product). The
    /// probe's tile is a slot of a ring written once and fenced before the
    /// loop; the slot an iteration reads comes from a parameter (zeros at
    /// run), so no load is invariant for the compiler to hoist out of an
    /// unrolled loop, and no barrier enters it.
    fn smem_byte(&self) -> Probe {
        let (op, tile, iters) = ("probe:smem_byte", Shape::new(32, 128), 1024);
        let units = (self.blocks * iters * tile.elems() * 4) as f64;
        self.make(op, ScalarDType::Float32, units, true, self.lowering(ROWS, false), |on| {
            let mut k = kernel(op, self.blocks, WARPS);
            let x = input::<F32>(&mut k, tile);
            let ring = on.then(|| {
                let ring = k.smem::<F32>("ring", 2 * tile.elems());
                for slot in 0..2 {
                    let s = k.smem_slot::<F32>(ring, slot, tile);
                    k.store(s, x);
                }
                (ring, k.param::<I32>("order", ParamKind::In, iters))
            });
            let zero = k.zeros::<F32>(tile);
            if on {
                k.barrier();
            }
            let [sum] = k.loop_(iters, [zero], |k, i, [sum]| {
                let y = match ring {
                    Some((ring, order)) => {
                        let slot = k.load_scalar(order, i) % 2;
                        let s = k.smem_slot::<F32>(ring, slot, tile);
                        k.load(s)
                    }
                    None => x,
                };
                [k.binary(sum, y, BinaryOp::Add)]
            });
            output(&mut k, self.blocks, sum);
            k.finish()
        })
    }

    /// [`STREAM`] bytes of i32 tiles, block-interleaved so the resident
    /// blocks read neighbouring tiles, each byte once.
    fn global_byte(&self) -> Probe {
        let (op, tile) = ("probe:global_byte", Shape::new(32, 128));
        let (blocks, iters) = self.stream_grid(tile.elems() * 4);
        self.make(op, ScalarDType::Int32, STREAM as f64, true, self.lowering(ROWS, false), |on| {
            let mut k = kernel(op, blocks, WARPS);
            let src = k.param::<I32>("src", ParamKind::In, STREAM / 4);
            let block = k.block(0);
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let sum = if on {
                sum_loads(&mut k, iters, zero, |k, i| {
                    k.view(src, (i * blocks + block) * tile.elems(), [tile.cols, 1], tile, [None, None])
                })
            } else {
                zero
            };
            output(&mut k, blocks, sum);
            k.finish()
        })
    }

    /// Four times [`STREAM`] bytes of [`CACHE_TILE`]s read by a single wave of
    /// blocks, every block resident from first to last, so the tiles they
    /// read an iteration together run on in one sweep, wrapped into a region
    /// of a prime number of tiles within `footprint` ([`region_tiles`]): a
    /// tile comes round again after a region of traffic, so a byte comes
    /// from wherever a region that size stays, and no two blocks read one
    /// tile together. Faster than DRAM, so it reads more than
    /// [`Self::global_byte`] to stand as clear of its baseline.
    fn cached(&self, op: &'static str, footprint: usize) -> Probe {
        let tile = CACHE_TILE;
        let o = self.target.occupancy;
        let blocks = (self.sms * o.simds * o.waves / WARPS) as usize;
        let iters = 4 * STREAM / (tile.elems() * 4) / blocks;
        let region = region_tiles(footprint);
        let units = (blocks * iters * tile.elems() * 4) as f64;
        self.make(op, ScalarDType::Int32, units, true, self.lowering(ROWS, false), |on| {
            let mut k = kernel(op, blocks, WARPS);
            let src = k.param::<I32>("src", ParamKind::In, region * tile.elems());
            let block = k.block(0);
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let sum = if on {
                sum_loads(&mut k, iters, zero, |k, i| {
                    let at = (i * blocks + block.clone()) % region * tile.elems();
                    k.view(src, at, [tile.cols, 1], tile, [None, None])
                })
            } else {
                zero
            };
            output(&mut k, blocks, sum);
            k.finish()
        })
    }

    /// A wave's bf16 row of 512 loaded each iteration and summed: the
    /// probe's view puts its columns 128 bytes apart, so a lane's every
    /// element is a 2-byte load of its own line, where the baseline's lanes
    /// read 16-byte runs. The rows cycle through a region every wave shares,
    /// so the lines come from cache either way.
    fn scatter(&self) -> Probe {
        let (op, cols, iters, region, span) = ("probe:scatter", 512, 2048, 8, 64 * 512);
        let tile = Shape::new(self.warps as usize, cols);
        let units = (iters * cols) as f64;
        self.make(op, ScalarDType::BFloat16, units, false, self.chain(Costs::default()), |on| {
            let mut k = kernel(op, self.sms as usize, self.warps);
            let src = k.param::<BF16>("src", ParamKind::In, region * span);
            let block = k.block(0);
            let zero = k.zeros::<BF16>(tile);
            let stride = if on { [1, span / cols] } else { [tile.cols, 1] };
            let sum = sum_loads(&mut k, iters, zero, |k, i| {
                k.view(src, (i + block) % region * span, stride, tile, [None, None])
            });
            output(&mut k, self.sms as usize, sum);
            k.finish()
        })
    }

    /// Blocks (a power of two filling the device) and iterations that read
    /// [`STREAM`] bytes in `tile_bytes` tiles.
    fn stream_grid(&self, tile_bytes: usize) -> (usize, usize) {
        let tiles = STREAM / tile_bytes;
        let blocks = self.blocks.next_power_of_two().min(tiles);
        (blocks, tiles / blocks)
    }

    /// 16×16×16 matrix-core steps on register operands into four
    /// independent accumulators per warp.
    fn step(&self) -> Probe {
        let lowering = self.lowering(WarpGrid { rows: 2, cols: 2 }, false);
        self.steps("probe:step", [64, 64, 16], self.blocks, WARPS, lowering, true)
    }

    /// The step of a wave alone on its SIMD, a block per SM of a wave per
    /// SIMD ([`Self::chain`]), each wave over the 32×32 accumulators a wave
    /// of [`Self::step`] holds: the chain's latency per step, where `step`
    /// is the core's throughput; their ratio is the waves a SIMD hides it
    /// with. A unit is one wave's step.
    fn step_lone(&self) -> Probe {
        let warps = self.warps;
        let tile = [32 * warps as usize, 32, 16];
        self.steps("probe:step_lone", tile, self.sms as usize, warps, self.chain(Costs::default()), false)
    }

    /// `blocks` of `warps` each accumulating a `bm × bn` product over `bk`
    /// every iteration, where the baseline holds the zero accumulator; a
    /// unit is a 16×16×16 step of the SM (`per_sm`) or of one wave.
    fn steps(
        &self,
        op: &'static str,
        [bm, bn, bk]: [usize; 3],
        blocks: usize,
        warps: u32,
        lowering: Lowering,
        per_sm: bool,
    ) -> Probe {
        let iters = 1024;
        let steps = iters * bm * bn * bk / 4096;
        let units = if per_sm { blocks * steps } else { steps / warps as usize } as f64;
        self.make(op, ScalarDType::BFloat16, units, per_sm, lowering, |on| {
            let mut k = kernel(op, blocks, warps);
            let a = input::<BF16>(&mut k, Shape::new(bm, bk));
            let b = input::<BF16>(&mut k, Shape::new(bn, bk));
            let zero = k.zeros::<F32>(Shape::new(bm, bn));
            let [acc] = if on {
                k.loop_(iters, [zero], |k, _, [acc]| [k.mma_oriented(acc, a, false, b, true, Some(Orient::Direct))])
            } else {
                [zero]
            };
            output(&mut k, blocks, acc);
            k.finish()
        })
    }

    /// A score tile's softmax vector work: scale, row max, subtract, `exp2`,
    /// row sum, the sum accumulated across iterations.
    fn score(&self) -> Probe {
        let (op, tile, iters) = ("probe:score", Shape::new(64, 128), 256);
        let units = (self.blocks * iters * tile.elems()) as f64;
        self.make(op, ScalarDType::Float32, units, true, self.lowering(ROWS, false), |on| {
            let mut k = kernel(op, self.blocks, WARPS);
            let x = input::<F32>(&mut k, tile);
            let zero = k.zeros::<F32>(Shape::new(tile.rows, 1));
            let [x, l] = if on {
                k.loop_(iters, [x, zero], |k, _, [x, l]| {
                    let scale = k.fill::<F32>(tile, Const::Float(0.5));
                    let s = k.binary(x, scale, BinaryOp::Mul);
                    let m = k.reduce(s, Axis::Row, ReduceOp::Max);
                    let d = k.binary(s, m, BinaryOp::Sub);
                    let e = k.unary(d, UnaryOp::Exp2);
                    let r = k.reduce(e, Axis::Row, ReduceOp::Sum);
                    [e, k.binary(l, r, BinaryOp::Add)]
                })
            } else {
                [x, zero]
            };
            output(&mut k, self.blocks, x);
            output(&mut k, self.blocks, l);
            k.finish()
        })
    }

    /// [`TRIPS`] trips of a pipeline over the least fill a block copies, a
    /// 16-byte chunk per thread (eight to a 32-wide i32 row), from a region
    /// every block shares so it stays in cache, lowered `unroll`ed (a trip's
    /// fixed cost) or rolled (slot offsets recomputed per trip). The baseline
    /// loads the same chunks into registers in a plain loop of as many trips,
    /// which the compiler reduces to little more than the loop (the region's
    /// chunks repeat), so the difference is a whole trip: the cached fill, the
    /// ring's store, fences and load. [`costs`] takes the load out.
    fn trips(&self, op: &'static str, unroll: bool) -> Probe {
        let (tile, region) = (trip_tile(self.target), 16);
        let stages = self.target.stages(2);
        let units = (self.blocks * TRIPS) as f64;
        self.make(op, ScalarDType::Int32, units, true, self.lowering(ROWS, unroll), |on| {
            // Both trip probes run these programs, named alike.
            let mut k = kernel("probe:trip", self.blocks, WARPS);
            let src = k.param::<I32>("src", ParamKind::In, region * tile.elems());
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let chunk =
                |k: &mut Kernel, step: Sc| k.view(src, step % region * tile.elems(), [32, 1], tile, [None, None]);
            let sum =
                if on { ring_sum(&mut k, TRIPS, stages, zero, chunk) } else { sum_loads(&mut k, TRIPS, zero, chunk) };
            output(&mut k, self.blocks, sum);
            k.finish()
        })
    }

    /// The half-wave exchanges of a relayout between two products: lanes
    /// trade a register with the lane 16 away, then with the neighbouring
    /// lane ([`half_swapped`]), so the value is the average of the moves a
    /// half-wave exchange issues ([`Costs::lane_move`]). Lowered with a free
    /// lane move and a dear round trip, so the inference exchanges.
    fn lane_move(&self) -> Probe {
        let plain = self.held();
        let helds = [half_swapped(&plain, 4), half_swapped(&plain, 0)];
        self.rehold("probe:lane_move", &plain, &helds, Costs { smem_trip: 1.0, ..Default::default() })
    }

    /// The re-holding across the rows of 16 lowered with a dear lane move:
    /// the round trip through shared memory, fenced both ways.
    fn smem_trip(&self) -> Probe {
        let plain = self.held();
        let helds = [half_swapped(&plain, 4)];
        self.rehold("probe:smem_trip", &plain, &helds, Costs { lane_move: 1.0, ..Default::default() })
    }

    /// How a wave's 16 × two-wave-widths bf16 tile (a score tile's
    /// probabilities) is held when nothing asks otherwise.
    fn held(&self) -> TileLayout {
        let tile = Shape::new(16 * self.warps as usize, 2 * self.target.wave as usize);
        natural(tile, self.warps, self.target.wave).expect("a natural layout")
    }

    /// A relayout-class pair: a bf16 tile carried as `plain` and, for each of
    /// `helds`, negated into a value pinned there and negated back, so the
    /// inference re-holds the tile there and back as `costs` choose; the
    /// baseline holds every value as `plain`, so its negations cancel and its
    /// loop is empty. A unit is the [`crate::layouts::Relayout::work`] of one
    /// wave's element re-held.
    fn rehold(&self, op: &'static str, plain: &TileLayout, helds: &[TileLayout], costs: Costs) -> Probe {
        let iters = 4096;
        let tile = plain.shape();
        let (warps, lanes) = (self.warps, self.target.wave);
        let work: f64 = helds
            .iter()
            .flat_map(|h| [plain.relayout(h, warps, lanes, &costs), h.relayout(plain, warps, lanes, &costs)])
            .map(|r| r.work())
            .sum();
        let units = iters as f64 * work * tile.elems() as f64 / warps as f64;
        self.make(op, ScalarDType::BFloat16, units, false, self.chain(costs), |on| {
            let mut k = kernel(op, self.sms as usize, warps);
            let x = input::<BF16>(&mut k, tile);
            let [x] = k.loop_(iters, [x], |k, _, [x]| {
                k.pin(x, plain.clone());
                [helds.iter().fold(x, |x, held| {
                    let y = k.unary(x, UnaryOp::Neg);
                    let next = k.unary(y, UnaryOp::Neg);
                    k.pin(y, if on { held.clone() } else { plain.clone() });
                    k.pin(next, plain.clone());
                    next
                })]
            });
            output(&mut k, self.sms as usize, x);
            k.finish()
        })
    }

    /// One wave per SM streaming a 16-byte chunk per lane through a
    /// two-stage ring, the waves' trips interleaved over [`STREAM`] so no
    /// line is read twice in a run: with nothing else resident and nothing
    /// to compute, every trip waits out its load on a nearly idle memory.
    /// The baseline is the ring's depth in trips. A lone wave's 16 bytes a
    /// lane are not bandwidth-bound: the difference is the wait alone.
    fn latency(&self) -> Probe {
        let op = "probe:latency";
        // A 16-byte chunk of i32 per lane.
        let tile = Shape::new(self.target.wave as usize, 4);
        let waves = self.sms as usize;
        let trips = LATENCY_TRIPS.min(STREAM / 4 / tile.elems() / waves);
        let stages = self.target.stages(2);
        let lowering = self.lowering(WarpGrid { rows: 1, cols: 1 }, false);
        self.make(op, ScalarDType::Int32, (trips - stages) as f64, false, lowering, |on| {
            let mut k = kernel(op, waves, 1);
            let src = k.param::<I32>("src", ParamKind::In, STREAM / 4);
            let block = k.block(0);
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let sum = ring_sum(&mut k, if on { trips } else { stages }, stages, zero, |k, step| {
                k.view(src, (step * waves + block) * tile.elems(), [4, 1], tile, [None, None])
            });
            output(&mut k, waves, sum);
            k.finish()
        })
    }
}

/// A kernel named after the probe `op`.
fn kernel(op: &str, blocks: usize, warps: u32) -> Kernel {
    let mut k = Kernel::new(op.replace(':', "_"));
    k.grid([Sc::from(blocks), Sc::from(1), Sc::from(1)]);
    k.warps(warps);
    k
}

/// `init` plus the tile `view` reads at each of `iters` iterations.
fn sum_loads<T: Elem>(
    k: &mut Kernel,
    iters: usize,
    init: Regs<T>,
    view: impl FnOnce(&mut Kernel, Sc) -> Gmem<T>,
) -> Regs<T> {
    let [sum] = k.loop_(iters, [init], |k, i, [sum]| {
        let v = view(k, i);
        let y = k.load(v);
        [k.binary(sum, y, BinaryOp::Add)]
    });
    sum
}

/// `init` plus the tile `view` reads at each of `trips` trips of a pipeline
/// staging it through a ring of `stages` slots.
fn ring_sum(
    k: &mut Kernel,
    trips: usize,
    stages: usize,
    init: Regs<I32>,
    view: impl FnOnce(&mut Kernel, Sc) -> Gmem<I32>,
) -> Regs<I32> {
    let tile = k.shape(init);
    let ring = k.smem::<I32>("ring", stages * tile.elems());
    let [sum] = k.pipeline(
        trips,
        stages,
        [init],
        |k, step, slot| {
            let v = view(k, step);
            let s = k.smem_slot::<I32>(ring, slot, tile);
            k.stage(s, v, CopyMode::Async);
        },
        |k, _, slot, [sum]| {
            let s = k.smem_slot::<I32>(ring, slot, tile);
            let y = k.load(s);
            [k.binary(sum, y, BinaryOp::Add)]
        },
    );
    sum
}

/// A `shape` tile loaded from a parameter of its own, so no value is known
/// to the compiler.
fn input<T: Elem>(k: &mut Kernel, shape: Shape) -> Regs<T> {
    let p = k.param::<T>("x", ParamKind::In, shape.elems());
    let v = k.view(p, 0, [shape.cols, 1], shape, [None, None]);
    k.load(v)
}

/// `v` stored to block `block(0)`'s slot of an output, so nothing is dead.
fn output<T: Elem>(k: &mut Kernel, blocks: usize, v: Regs<T>) {
    let shape = k.shape(v);
    let p = k.param::<T>("y", ParamKind::Out, blocks * shape.elems());
    let block = k.block(0);
    let view = k.view(p, block * shape.elems(), [shape.cols, 1], shape, [None, None]);
    k.store(view, v);
}

/// `l` with its first register bit and lane bit `lane` exchanged: each lane
/// keeps half its elements and takes the other half from lane `lane ^ 2^bit`.
pub(crate) fn half_swapped(l: &TileLayout, lane: u32) -> TileLayout {
    let bases = |dim| -> Vec<[u32; 2]> {
        (0..l.frag.in_bits(dim))
            .map(|b| {
                let get = |out| l.frag.basis(dim, b).iter().find(|o| o.0 == out).map_or(0, |o| o.1);
                [get(Dim::Row), get(Dim::Col)]
            })
            .collect()
    };
    let (mut reg, mut lanes) = (bases(Dim::Reg), bases(Dim::Lane));
    std::mem::swap(&mut reg[0], &mut lanes[lane as usize]);
    let outs = [(Dim::Row, l.frag.out_size(Dim::Row)), (Dim::Col, l.frag.out_size(Dim::Col))];
    TileLayout { frag: Layout::from_bases(outs, &[(Dim::Reg, &reg), (Dim::Lane, &lanes)]), ..l.clone() }
}
