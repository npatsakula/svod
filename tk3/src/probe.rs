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

/// Bytes the streaming probes read once per run: larger than any device
/// cache (MI300X's Infinity Cache is 256 MB), so every run streams from DRAM.
pub(crate) const STREAM: usize = 512 << 20;
/// Trips of a lone wave at most: a few ms of waiting.
const LATENCY_TRIPS: usize = 16 << 10;
/// Trips of the trip probes: enough that their fixed costs dwarf a launch.
const TRIPS: usize = 4096;
/// Timings of a pair, compiled once, before a probe gives up: one with a
/// weak signal ([`strong`]) or a lost stamp is timed again.
const TRIES: usize = 3;
/// Warps of a throughput probe's block.
const WARPS: u32 = 4;
/// A throughput probe's warp grid: [`WARPS`] rows of warps.
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

/// The costs of the default device, whose target is `target`: every field
/// measured once per device, compiler and probe program, then read from the
/// tune store.
pub fn costs(target: &Target) -> Result<Costs> {
    let probes = probes(target)?;
    let compiler = tune::compiler(&default_device());
    let mut costs = Costs::default();
    let fields = costs.fields();
    assert_eq!(fields.len(), probes.len(), "a probe per field");
    for ((name, value), probe) in fields.into_iter().zip(probes) {
        assert_eq!(probe.op.strip_prefix("probe:"), Some(name), "probes in field order");
        *value = measured(target, compiler, probe)?;
    }
    // A plan prices a trip's fragment reads by `smem_byte`, so a trip is
    // what its pipeline adds besides reading its tile from the ring.
    let fragment = (trip_tile(target).elems() * 4) as f64 * costs.smem_byte;
    costs.trip -= fragment;
    costs.trip_rolled -= fragment;
    Ok(costs)
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
            (0..TRIES).find_map(|_| timed(&mut built, scale / units).map_err(|e| last = e).ok())
        },
    );
    Ok(fs.context(UnmeasuredSnafu { probe: op, last })? as f64 * 1e-6)
}

/// One timing of a compiled pair: its difference times `per`, in
/// femtoseconds, or why it does not hold.
fn timed(built: &mut [Built], per: f64) -> Result<u64, String> {
    match tune::time(built)[..] {
        [Some(base_ns), Some(probe_ns)] if strong(base_ns, probe_ns) => {
            Ok(((probe_ns - base_ns) as f64 * per * 1e6).round() as u64)
        }
        [Some(base_ns), Some(probe_ns)] => {
            let signal = (probe_ns as f64 - base_ns as f64) / probe_ns as f64;
            Err(format!("a signal of {signal:.3} of the probe's time (baseline {base_ns} ns, probe {probe_ns} ns)"))
        }
        _ => Err("a program failed to build or lost its stamps".into()),
    }
}

/// Every probe of `target`, in [`Costs::fields`] order.
pub(crate) fn probes(target: &Target) -> Result<Vec<Probe>> {
    let p = Probes::new(target)?;
    Ok(vec![
        p.smem_byte(),
        p.global_byte(),
        p.scatter(),
        p.step(),
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
        let (op, [bm, bn, bk], iters) = ("probe:step", [64, 64, 16], 1024);
        let units = (self.blocks * iters * bm * bn * bk / 4096) as f64;
        let lowering = self.lowering(WarpGrid { rows: 2, cols: 2 }, false);
        self.make(op, ScalarDType::BFloat16, units, true, lowering, |on| {
            let mut k = kernel(op, self.blocks, WARPS);
            let a = input::<BF16>(&mut k, Shape::new(bm, bk));
            let b = input::<BF16>(&mut k, Shape::new(bn, bk));
            let zero = k.zeros::<F32>(Shape::new(bm, bn));
            let [acc] = if on {
                k.loop_(iters, [zero], |k, _, [acc]| [k.mma_oriented(acc, a, false, b, true, Some(Orient::Direct))])
            } else {
                [zero]
            };
            output(&mut k, self.blocks, acc);
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
