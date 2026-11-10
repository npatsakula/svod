//! `c = scale·act(a·bᵀ + bias) + residual` (or `scale·act(gate)·up` for a
//! gated weight) as a K-streaming block pipeline with the epilogue on the f32
//! accumulator.

use std::fmt;
use std::num::NonZeroUsize;

use super::{Act, Admits, Admitted, Batch, Spec, batch_offset, bound, konst, load_f32};
use crate::atoms::{Core, Device};
use crate::build::*;
use crate::ir::*;
use crate::lower::Lowering;
use crate::ops::config::{GemmProblem, Planner, Rejected};

/// What runs on the accumulator before the single rounding at the store.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Epilogue {
    /// A `[n]` (`[2n]` gated) bias parameter.
    pub bias: bool,
    pub act: Act,
    /// The weight is `[2n, k]`, gate rows over up rows; the output is
    /// `act(gate)·up` (SwiGLU with `Silu`, GeGLU with `Gelu`).
    pub gated: bool,
    /// A `[m, n]` residual parameter added last.
    pub residual: bool,
    /// Multiplies the activated (gated) value, before the residual add.
    pub scale: Option<Scale>,
    /// The output (and residual) is f32: the accumulator is stored unrounded.
    pub out_f32: bool,
    /// Rotates the output's first heads after the bias, alone in an epilogue.
    pub rope: Option<Rope>,
}

impl Epilogue {
    pub const DEFAULT: Self =
        Self { bias: false, act: Act::None, gated: false, residual: false, scale: None, out_f32: false, rope: None };
}

/// A rotary embedding of the output's first `heads` heads of `head_dim`
/// columns by `[seq, head_dim / 2]` cos and sin tables, output row `m` at
/// position `m % seq`: a head's halves turn together as `apply_rotary_emb`
/// turns split halves, `x₁·cos − x₂·sin` and `x₂·cos + x₁·sin`. The columns
/// after them (a fused projection's values) are stored as they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rope {
    pub heads: usize,
    pub head_dim: usize,
    pub seq: usize,
}

/// A [`Rope`] with its cos and sin tables.
type Turn<'k, T> = (Rope, ParamRef<'k, T, In>, ParamRef<'k, T, In>);

/// The tune store keys by this text: an epilogue without a scale prints as
/// it did before the field existed, so its stored choices still apply.
impl fmt::Debug for Epilogue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Epilogue");
        s.field("bias", &self.bias).field("act", &self.act).field("gated", &self.gated);
        s.field("residual", &self.residual);
        if let Some(scale) = self.scale {
            s.field("scale", &scale);
        }
        if self.out_f32 {
            s.field("out_f32", &true);
        }
        if let Some(rope) = self.rope {
            s.field("rope", &rope);
        }
        s.finish()
    }
}

/// An f32 output scale, compared and hashed by its bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scale(u32);

impl Scale {
    pub const fn new(value: f32) -> Self {
        Self(value.to_bits())
    }

    pub fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl fmt::Debug for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

/// The tune store keys by this text ([`Schedule`]'s included, the
/// [`TileOrder`] only off its default).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmCfg {
    /// `[bm, bn, bk]`.
    pub tile: [usize; 3],
    /// `[rows, cols]` of the warp grid over the output tile.
    pub warps: [u32; 2],
    pub core: Core,
    /// The ring: one crew over `depth` slots, or the two warp rows as crews
    /// a cluster apart over one slot, the operands k-sliced a product deep.
    pub schedule: Schedule,
    /// The raster the blocks walk the output tiles in: the planner plans
    /// [`TileOrder::DEFAULT`], a config may carry another.
    pub order: TileOrder,
    /// The last, partial wave of tiles split along K ([`TailPlan`]); every
    /// tile data-parallel where `None`.
    pub tail: Option<TailSplit>,
}

/// A config of the default raster and no tail split prints as it did
/// before those fields existed, so a ranking's text moves only where they
/// do.
impl fmt::Debug for GemmCfg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("GemmCfg");
        s.field("tile", &self.tile).field("warps", &self.warps).field("core", &self.core);
        s.field("schedule", &self.schedule);
        if self.order != TileOrder::DEFAULT {
            s.field("order", &self.order);
        }
        if let Some(tail) = self.tail {
            s.field("tail", &tail);
        }
        s.finish()
    }
}

/// How a GEMM's last, partial wave of output tiles runs: each of its tiles
/// over `split` blocks, a share of the reduction each ([`TailPlan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TailSplit {
    split: usize,
}

impl TailSplit {
    /// A tail split over `split` blocks a tile; `None` below two.
    pub const fn new(split: usize) -> Option<Self> {
        if split >= 2 { Some(Self { split }) } else { None }
    }

    pub const fn split(self) -> usize {
        self.split
    }
}

/// The raster blocks walk the output tiles in: groups of `rows` tile rows,
/// a group's blocks down its rows a column at a time, so the blocks in
/// flight share their B tiles (a GEMM's weight, a split nearest kernel's
/// candidate shares) and a group's A rows stay in the cache while it sweeps
/// the columns. One row is row-major; the grid's rows or more,
/// column-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileOrder {
    rows: NonZeroUsize,
}

impl TileOrder {
    /// Groups of eight tile rows: the raster every planned GEMM and every
    /// nearest kernel walks; another is a config's own (a hand-made code's).
    pub const DEFAULT: Self = Self::new(NonZeroUsize::new(8).unwrap());

    /// The tallest group, a column-major walk of any grid of fewer tile
    /// rows: a taller one (a hand-made code's) is clamped to it, so a
    /// group's span of tiles (`rows · gn`) stays far inside the scalars'
    /// range.
    pub const MOST_ROWS: usize = 1 << 15;

    /// Groups of `rows` tile rows, at most [`Self::MOST_ROWS`].
    pub const fn new(rows: NonZeroUsize) -> Self {
        if rows.get() > Self::MOST_ROWS {
            Self { rows: NonZeroUsize::new(Self::MOST_ROWS).unwrap() }
        } else {
            Self { rows }
        }
    }

    /// Groups of `rows` tile rows; `None` for none.
    pub const fn grouped(rows: usize) -> Option<Self> {
        match NonZeroUsize::new(rows) {
            Some(rows) => Some(Self::new(rows)),
            None => None,
        }
    }

    pub const fn rows(self) -> usize {
        self.rows.get()
    }

    /// The rows a group spans in a grid of `gm` tile rows.
    pub fn group(self, gm: usize) -> usize {
        self.rows().min(gm)
    }

    /// The `(row, col)` tile of this block in a grid of `gm × gn` tiles
    /// (grid x and y), walking groups of [`Self::rows`] tile rows.
    pub(crate) fn tile<'k>(self, k: &mut Kernel<'k>, gm: impl Into<Sc<'k>>, gn: usize) -> (Sc<'k>, Sc<'k>) {
        let gm = gm.into();
        let id = k.block(0) + k.block(1) * gm.clone();
        self.walk(id, gm, gn)
    }

    /// The `(row, col)` tile the walk of a `gm × gn` grid reaches `id`th.
    pub(crate) fn walk<'k>(self, id: Sc<'k>, gm: Sc<'k>, gn: usize) -> (Sc<'k>, Sc<'k>) {
        let group = self.rows();
        let first_m = id.clone() / (group * gn) * group;
        let rows = (gm - first_m.clone()).min(group);
        let within = id % (group * gn);
        (first_m + within.clone() % rows.clone(), within / rows)
    }
}

/// A run of a raster's walk: `count` output tiles from the walk's `first`th,
/// a block each down grid x.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Run {
    first: usize,
    count: NonZeroUsize,
}

impl Run {
    /// `None` for no tiles.
    pub const fn new(first: usize, count: usize) -> Option<Self> {
        match NonZeroUsize::new(count) {
            Some(count) => Some(Self { first, count }),
            None => None,
        }
    }

    pub const fn first(self) -> usize {
        self.first
    }

    pub const fn count(self) -> usize {
        self.count.get()
    }

    /// The walk's indices the run covers.
    pub fn range(self) -> std::ops::Range<usize> {
        self.first..self.first + self.count()
    }
}

/// The output tiles a GEMM launch covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tiles {
    /// Every tile, a block each over grid x and y.
    Grid,
    /// A run of the raster's walk.
    Run(Run),
}

impl Tiles {
    /// Grid x and y over a `gm × gn` grid of tiles.
    pub(crate) fn grid<'k>(self, gm: usize, gn: usize) -> [Sc<'k>; 2] {
        match self {
            Tiles::Grid => [Sc::from(gm), Sc::from(gn)],
            Tiles::Run(run) => [Sc::from(run.count()), Sc::from(1)],
        }
    }

    /// The `(row, col)` tile of this block under `order`.
    pub(crate) fn tile<'k>(self, k: &mut Kernel<'k>, order: TileOrder, gm: usize, gn: usize) -> (Sc<'k>, Sc<'k>) {
        match self {
            Tiles::Grid => order.tile(k, gm, gn),
            Tiles::Run(run) => {
                let id = k.block(0) + run.first;
                order.walk(id, Sc::from(gm), gn)
            }
        }
    }
}

/// A GEMM of a single batch whose tiles take more than a wave of the blocks
/// a device runs at once, in whole waves data-parallel and the rest — the
/// last, partial wave — each tile over `split` blocks a share of the
/// reduction long: the waves' GEMM, the tail's f32 partials
/// (`[split, tail tiles, bm, bn]`, a tile's elements row-major) and their
/// [`split_merge`] under the epilogue, three launches over runs of one
/// raster's walk. Fewer tiles than a wave split whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TailPlan {
    /// The whole waves' tiles, from the walk's first; `None` short of a wave.
    pub waves: Option<Run>,
    /// The rest of the walk.
    pub tail: Run,
    pub split: usize,
}

impl TailPlan {
    /// The plan of `tiles` tiles on a device running `wave` blocks at once,
    /// the tail `split`; `None` where whole waves leave no tail.
    pub fn new(tiles: usize, wave: usize, split: TailSplit) -> Option<Self> {
        let full = tiles / wave.max(1) * wave.max(1);
        let tail = Run::new(full, tiles - full)?;
        Some(Self { waves: Run::new(0, full), tail, split: split.split() })
    }

    /// The plan whose every tile splits: a whole-problem split-K over the
    /// same launches.
    #[cfg(test)]
    pub(crate) fn whole(tiles: usize, split: usize) -> Option<Self> {
        Some(Self { waves: None, tail: Run::new(0, tiles)?, split })
    }
}

impl Default for TileOrder {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl GemmCfg {
    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering { core: self.core, ..Lowering::new(device) }
    }

    pub fn stages(&self) -> usize {
        self.schedule.stages()
    }

    pub fn unroll(&self) -> bool {
        self.schedule.unroll()
    }

    pub fn crews(&self) -> u32 {
        self.schedule.crews()
    }

    /// The k-slices each operand of a slot is held as, one after another:
    /// under two crews one matrix-core step deep, so a slice is a plain view
    /// and a product's fragments are one cluster's reads; else the slot whole.
    pub fn slices(&self) -> usize {
        self.schedule.kslice().map_or(1, |k| self.tile[2] / k)
    }

    /// Whether the tile carries an [`Epilogue::rope`] of `head_dim` heads:
    /// the block's columns and each warp's hold whole heads, so a lane holds
    /// both halves of its heads and the turn moves no data between lanes.
    pub fn turns(&self, head_dim: usize) -> bool {
        let bn = self.tile[1];
        let warp_cols = bn / self.warps[1] as usize;
        (head_dim / 2).is_power_of_two() && bn.is_multiple_of(head_dim) && warp_cols.is_multiple_of(head_dim)
    }

    /// Static shared memory of the operand ring over elements of `bytes`
    /// bytes: what [`mainloop`] allocates, a slot of the A tile and of
    /// each B tile (two gated) per stage.
    pub fn smem_bytes(&self, gated: bool, bytes: usize) -> usize {
        let [bm, bn, bk] = self.tile;
        self.stages() * (bm + bn * if gated { 2 } else { 1 }) * bk * bytes
    }
}

/// `m` rows per batch; the batch, when there is one, walks grid z. A
/// kernel runs it under a config admitted for it ([`GemmSpec::admitted`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: Batch,
    pub epilogue: Epilogue,
}

impl Spec for GemmSpec {
    type Cfg = GemmCfg;
}

impl Admits for GemmSpec {
    type Problem = GemmProblem;

    fn problem(&self) -> GemmProblem {
        GemmProblem::new(self.batch.capacity().max(1), self.m, self.n, self.k, &self.epilogue)
    }
}

impl GemmSpec {
    /// `cfg` [admitted](Admitted) for this GEMM on `planner`, or why it is not.
    pub fn admitted(self, planner: &Planner, cfg: GemmCfg) -> Result<Admitted<'_, Self>, Rejected> {
        self.admit(planner, cfg)
    }
}

/// Parameters in order: `a [batch·m, k]`, `b [n, k]` (`[2n, k]` gated),
/// `bias` if any, `residual [batch·m, n]` if any, the `cos` and `sin`
/// tables of a [`Rope`], `c [batch·m, n]`. Rows past `m` and weight rows
/// past `n` are clamped reads whose results are never stored; `k` must be a
/// multiple of `bk`. Over 8-bit integers the i32 accumulator is widened to
/// f32 for the epilogue, which stores f32 (`out_f32`): exact while the sums
/// stay below 2^24.
pub fn gemm<T: MmaIn>(spec: &Admitted<GemmSpec>) -> Program {
    gemm_over::<T>(spec, Tiles::Grid, 1)
}

/// [`gemm`] over `tiles`, split `split` ways along K as [`mainloop_gemm`]
/// splits it.
fn gemm_over<T: MmaIn>(spec: &Admitted<GemmSpec>, tiles: Tiles, split: usize) -> Program {
    let GemmSpec { m, k: kk, .. } = **spec;
    let [bm, _, bk] = spec.cfg().tile;
    mainloop_gemm::<T>("gemm", spec, m * kk, tiles, split, |k, a, bb, row0| {
        let a_view = k.view(a, batch_offset(bb, m * kk), [kk, 1], Shape::new(bm, bk), [bound(m, bm), None]);
        let a_view = k.at(a_view, row0, 0);
        Box::new(move |k, _step, koff| k.at(a_view, 0, koff))
    })
}

/// A GEMM run under a [`TailPlan`]: the spec's config, raster and
/// epilogue, a single static batch, neither gated nor turned.
#[derive(Clone, Debug, PartialEq)]
pub struct TailSpec<'t> {
    pub gemm: Admitted<'t, GemmSpec>,
    pub plan: TailPlan,
}

/// Search bound: the tile of a tail's merge, f32 rows in natural layouts:
/// the conv merge's, a sub-tile of the GEMM's (whose columns are then a
/// multiple of 32).
pub(crate) const TAIL_MERGE: [usize; 2] = [32, 32];

impl TailSpec<'_> {
    /// Whether a GEMM of `m × n` over `cfg`'s tiles merges a tail: the
    /// merge tile divides the GEMM's.
    pub fn merges(cfg: &GemmCfg) -> bool {
        cfg.tile[0].is_multiple_of(TAIL_MERGE[0]) && cfg.tile[1].is_multiple_of(TAIL_MERGE[1])
    }

    /// The f32 partials' elements: a `[bm, bn]` slot a tail tile a split.
    pub fn partials(&self) -> usize {
        let [bm, bn, _] = self.gemm.cfg().tile;
        self.plan.split * self.plan.tail.count() * bm * bn
    }

    /// The merge of the tail's partials.
    pub fn merge(&self) -> SplitMergeSpec {
        let GemmSpec { m, n, epilogue, .. } = *self.gemm;
        let cfg = self.gemm.cfg();
        let parts = Parts { order: cfg.order, tile: [cfg.tile[0], cfg.tile[1]], run: self.plan.tail };
        SplitMergeSpec { m, n, split: self.plan.split, epilogue, tile: TAIL_MERGE, parts: Some(parts) }
    }
}

/// A [`TailPlan`]'s programs in launch order: the whole waves' GEMM where
/// there are any (parameters as [`gemm`]'s), the tail's partials (`a`,
/// `b`, the f32 partials [`TailSpec::partials`] long) and their merge
/// ([`split_merge`]'s parameters).
pub fn gemm_tail<T: MmaIn>(spec: &TailSpec) -> Vec<Program> {
    let TailSpec { gemm, plan } = spec;
    let GemmSpec { ref batch, epilogue, k, .. } = **gemm;
    let cfg = gemm.cfg();
    assert!(
        *batch == Batch::Static(1) && !epilogue.gated && epilogue.rope.is_none(),
        "{epilogue:?} over {batch:?} splits no tail"
    );
    assert!(plan.split > 1 && (k / cfg.tile[2]).is_multiple_of(plan.split), "{plan:?} divides the trips");
    let waves = plan.waves.map(|run| gemm_over::<T>(gemm, Tiles::Run(run), 1));
    let tail = gemm_over::<T>(gemm, Tiles::Run(plan.tail), plan.split);
    waves.into_iter().chain([tail, split_merge::<T>(&spec.merge())]).collect()
}

/// The GEMM around an A operand of `a_elems` elements per batch whose
/// K-step tiles `a` describes: given the kernel, the parameter, the batch
/// index and the block's first row, it returns the source of the `[bm, bk]`
/// tile at `(step, step·bk)`, over the output `tiles`. The output dtype
/// follows the epilogue.
///
/// With `split > 1` (a single static batch) grid z splits the reduction:
/// split `z` walks steps `z·trips/split..`, and the kernel writes its raw f32
/// accumulator for [`split_merge`] to finish: to `c [split, m, n]` over the
/// whole grid, to a `[bm, bn]` slot a tile of `c [split, count, bm, bn]`
/// over a run of `count` tiles: of a tile across the output's edge, only
/// the slot's elements inside it.
pub(crate) fn mainloop_gemm<T: MmaIn>(
    name: &str,
    spec: &Admitted<GemmSpec>,
    a_elems: usize,
    tiles: Tiles,
    split: usize,
    a: impl for<'k> FnOnce(&mut Kernel<'k>, ParamRef<'k, T, In>, &Option<Sc<'k>>, Sc<'k>) -> ASource<'k, T>,
) -> Program {
    assert!(T::DTYPE.bytes() == 2 || spec.epilogue.out_f32 || split > 1, "an 8-bit product stores f32");
    if split > 1 {
        Kernel::record(name, |k| build::<T, F32>(k, spec, a_elems, tiles, split, a))
    } else if spec.epilogue.out_f32 {
        Kernel::record(name, |k| build::<T, F32>(k, spec, a_elems, tiles, 1, a))
    } else {
        Kernel::record(name, |k| build::<T, T>(k, spec, a_elems, tiles, 1, a))
    }
}

/// The source of the `[bm, bk]` A tile of a step: given the kernel, the step
/// and its column offset.
pub(crate) type ASource<'k, T> = Box<dyn Fn(&mut Kernel<'k>, Sc<'k>, Sc<'k>) -> Gmem<'k, T> + 'k>;

fn build<'k, T: MmaIn, U: Elem>(
    k: &mut Kernel<'k>,
    spec: &Admitted<GemmSpec>,
    a_elems: usize,
    tiles: Tiles,
    split: usize,
    a_source: impl FnOnce(&mut Kernel<'k>, ParamRef<'k, T, In>, &Option<Sc<'k>>, Sc<'k>) -> ASource<'k, T>,
) {
    let GemmSpec { m, n, k: kk, ref batch, epilogue } = **spec;
    let cfg = spec.cfg();
    let [bm, bn, bk] = cfg.tile;
    let trips = kk / bk;
    let partial = split > 1;
    assert!(!partial || (*batch == Batch::Static(1) && !epilogue.gated && trips.is_multiple_of(split)));
    // A partial kernel stores the bare accumulator.
    let epi = if partial { Epilogue::DEFAULT } else { epilogue };
    let halves = if epi.gated { 2 } else { 1 };
    let cap = batch.capacity();
    let a = k.input::<T>("a", cap * a_elems);
    let b = k.input::<T>("b", halves * n * kk);
    let bias = epi.bias.then(|| k.input::<T>("bias", halves * n));
    let residual = epi.residual.then(|| k.input::<U>("residual", cap * m * n));
    let rope = epi.rope.map(|rope| {
        let plain = Epilogue { bias: epi.bias, rope: epi.rope, ..Epilogue::DEFAULT };
        assert!(epi == plain && rope.heads * rope.head_dim <= n, "{epi:?} turns no more than a bias");
        let table = rope.seq * rope.head_dim / 2;
        (rope, k.input::<T>("cos", table), k.input::<T>("sin", table))
    });
    // A run's partials take a slot a tile; the whole grid's, the output's shape.
    let slots = match tiles {
        Tiles::Run(run) if partial => Some(run.count()),
        _ => None,
    };
    let c = k.output::<U>("c", slots.map_or(cap.max(split) * m * n, |count| split * count * bm * bn));
    let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
    let (gz, bb) = if partial { (Sc::from(split), Some(k.block(2))) } else { batch.axis(k) };
    let [gx, gy] = tiles.grid(gm, gn);
    k.grid([gx, gy, gz]);
    k.warps(WarpGrid { rows: cfg.warps[0], cols: cfg.warps[1] });

    let (bx, by) = tiles.tile(k, cfg.order, gm, gn);
    let (row0, col0) = (bx * bm, by * bn);
    let n_bound = bound(n, bn);
    let a_tile = a_source(k, a, if partial { &None } else { &bb }, row0.clone());
    let b_view = |k: &mut Kernel<'k>, half: usize| {
        let v = k.view(b, half * n * kk, [kk, 1], Shape::new(bn, bk), [n_bound.clone(), None]);
        k.at(v, col0.clone(), 0)
    };
    let schedule = cfg.schedule;
    let first_step = partial.then(|| bb.clone().expect("the split index") * (trips / split));
    let sa = Shape::new(bm, bk);
    let accs = if epi.gated {
        let halves = [b_view(k, 0), b_view(k, 1)];
        Vec::from(mainloop(k, sa, &a_tile, halves, trips / split, schedule, bk, first_step, cfg.slices()))
    } else {
        let whole = [b_view(k, 0)];
        Vec::from(mainloop(k, sa, &a_tile, whole, trips / split, schedule, bk, first_step, cfg.slices()))
    };
    let accs = accs.into_iter().map(|acc| T::widened(k, acc)).collect();
    let tile = Tile2 { m, n, bm, bn, row0, col0, bounds: [bound(m, bm), bound(n, bn)] };
    let dest = match slots {
        Some(count) => {
            let slot = bb.clone().expect("the split index") * count + k.block(0);
            Dest::Slot(c, slot * (bm * bn))
        }
        None => Dest::Tile(c),
    };
    finish::<T, U>(k, accs, epi, bias, residual, rope, dest, &bb, &tile);
}

/// Where a block's `[bm, bn]` output tile sits in an `[m, n]` output, and
/// the bounds its rows and columns are cut at (none where no tile of its
/// grid crosses the edge).
struct Tile2<'k> {
    m: usize,
    n: usize,
    bm: usize,
    bn: usize,
    row0: Sc<'k>,
    col0: Sc<'k>,
    bounds: [Option<Sc<'k>>; 2],
}

impl<'k> Tile2<'k> {
    /// The bounds measured from the tile's corner: the rows and columns of
    /// its `[bm, bn]` that lie in the `[m, n]` output.
    fn within(&self) -> [Option<Sc<'k>>; 2] {
        let [rows, cols] = self.bounds.clone();
        [rows.map(|b| b - self.row0.clone()), cols.map(|b| b - self.col0.clone())]
    }
}

/// Where the epilogue stores a block's tile.
enum Dest<'k, U> {
    /// Its window of the `[m, n]` output.
    Tile(ParamRef<'k, U, Out>),
    /// A `[bm, bn]` slot of its own, row-major from this element.
    Slot(ParamRef<'k, U, Out>, Sc<'k>),
}

/// The epilogue on the f32 accumulators (two when gated) and the store.
#[allow(clippy::too_many_arguments)]
fn finish<'k, T: Elem, U: Elem>(
    k: &mut Kernel<'k>,
    accs: Vec<Regs<'k, F32>>,
    epi: Epilogue,
    bias: Option<ParamRef<'k, T, In>>,
    residual: Option<ParamRef<'k, U, In>>,
    rope: Option<Turn<'k, T>>,
    c: Dest<'k, U>,
    bb: &Option<Sc<'k>>,
    t: &Tile2<'k>,
) {
    let Tile2 { n, bn, ref col0, ref bounds, .. } = *t;
    let n_bound = bounds[1].clone();
    let mut accs = accs.into_iter().enumerate().map(|(half, acc)| match bias {
        Some(bias) => {
            let v = k.view(bias, half * n, [0, 1], Shape::new(1, bn), [None, n_bound.clone()]);
            let v = k.at(v, 0, col0.clone());
            let v = load_f32(k, v);
            k.binary(acc, v, BinaryOp::Add)
        }
        None => acc,
    });
    let first = accs.next().expect("an accumulator");
    let up = accs.next();
    let mut out = epi.act.apply(k, first);
    if let Some(up) = up {
        out = k.binary(out, up, BinaryOp::Mul);
    }
    if let Some(scale) = epi.scale {
        let s = konst(k, out, f64::from(scale.get()));
        out = k.binary(out, s, BinaryOp::Mul);
    }
    if let Some((rope, cos, sin)) = rope {
        out = turn(k, out, rope, cos, sin, t);
    }
    if let Some(residual) = residual {
        let r = tile(k, residual, bb, t);
        let r = load_f32(k, r);
        out = k.binary(out, r, BinaryOp::Add);
    }
    let out = k.cast::<F32, U>(out);
    let c_view = match c {
        Dest::Tile(c) => tile(k, c, bb, t),
        Dest::Slot(c, at) => k.view(c, at, [bn, 1], Shape::new(t.bm, bn), t.within()),
    };
    k.store(c_view, out);
}

/// The block's `[bm, bn]` window of an `[m, n]` parameter.
fn tile<'k, U: Elem, A: Access>(
    k: &mut Kernel<'k>,
    p: ParamRef<'k, U, A>,
    bb: &Option<Sc<'k>>,
    t: &Tile2<'k>,
) -> Gmem<'k, U, A> {
    let Tile2 { m, n, bm, bn, ref row0, ref col0, ref bounds } = *t;
    let v = k.view(p, batch_offset(bb, m * n), [n, 1], Shape::new(bm, bn), bounds.clone());
    k.at(v, row0.clone(), col0.clone())
}

/// [`Epilogue::rope`] on the output tile `x`, in a tile holding rotated
/// columns: `x·cos + x'·sin`, `x'` the element half a head along
/// ([`Kernel::flip`]) and `sin` negated over a head's first half.
/// A tile across the last rotated head keeps `x` past it.
fn turn<'k, T: Elem>(
    k: &mut Kernel<'k>,
    x: Regs<'k, F32>,
    rope: Rope,
    cos: ParamRef<'k, T, In>,
    sin: ParamRef<'k, T, In>,
    t: &Tile2<'k>,
) -> Regs<'k, F32> {
    let Tile2 { bm, bn, ref row0, ref col0, .. } = *t;
    let (half, turned) = (rope.head_dim / 2, rope.heads * rope.head_dim);
    let [y] = k.select_if(
        col0.clone().lt(turned),
        |k| {
            // Every head of row `r` reads the table row of its position.
            let table = |k: &mut Kernel<'k>, p: ParamRef<'k, T, In>| {
                let at = (row0.clone() + k.row()) % rope.seq * half;
                let v = k.gather(p, 0, at, None, Shape::new(bm, bn), None);
                let v = k.periodic(v, half);
                load_f32(k, v)
            };
            let (c, s) = (table(k, cos), table(k, sin));
            let cols = Shape::new(1, bn);
            let col = k.coord(cols, Axis::Col);
            let int = |k: &mut Kernel<'k>, v: Sc<'k>| k.splat::<I32>(cols, v);
            // A column of a head's first half: `c / half` is twice `c / head_dim`.
            let (halves, heads, two) = (int(k, Sc::from(half)), int(k, Sc::from(rope.head_dim)), int(k, Sc::from(2)));
            let in_half = k.binary(col, halves, BinaryOp::Div);
            let head = k.binary(col, heads, BinaryOp::Div);
            let twice = k.binary(head, two, BinaryOp::Mul);
            let first = k.compare(in_half, twice, CmpOp::Eq);
            let (minus, plus) = (k.fill::<F32>(cols, Const::Float(-1.0)), k.fill::<F32>(cols, Const::Float(1.0)));
            let sign = k.where_(first, minus, plus);
            let s = k.binary(s, sign, BinaryOp::Mul);
            let partner = k.flip(x, Axis::Col, half as u32);
            let xc = k.binary(x, c, BinaryOp::Mul);
            let y = k.mul_add(partner, s, xc);
            if turned.is_multiple_of(bn) {
                return [y];
            }
            let whole = Shape::new(bm, bn);
            let at = k.coord(whole, Axis::Col);
            let end = k.splat::<I32>(whole, Sc::from(turned) - col0.clone());
            let inside = k.compare(at, end, CmpOp::Lt);
            [k.where_(inside, y, x)]
        },
        |_| [x],
    );
    y
}

/// The second program of a split reduction: `c = epilogue(Σ_z partial[z])`
/// over f32 partials, `bm × bn` tiles per block: over the whole output from
/// `partial [split, m, n]`, or over the tiles of a GEMM's run ([`Parts`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SplitMergeSpec {
    pub m: usize,
    pub n: usize,
    pub split: usize,
    pub epilogue: Epilogue,
    /// `[bm, bn]`, powers of two (dividing the [`Parts`]' tile).
    pub tile: [usize; 2],
    pub parts: Option<Parts>,
}

/// The output tiles a run of a GEMM's raster partials holds, a
/// `[tile]` slot of each per split: `partial [split, run count, tile]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Parts {
    pub order: TileOrder,
    /// The GEMM's `[bm, bn]`.
    pub tile: [usize; 2],
    pub run: Run,
}

/// Parameters in order: `partial` f32 (`[split, m, n]`, or a [`Parts`]'
/// slots), `bias [n]` if any, `residual [m, n]` if any, `c [m, n]` (f32
/// when `out_f32`). Over a run's parts only the run's tiles of `c` are
/// written.
pub fn split_merge<T: Elem>(spec: &SplitMergeSpec) -> Program {
    if spec.epilogue.out_f32 { merge::<T, F32>(spec) } else { merge::<T, T>(spec) }
}

fn merge<T: Elem, U: Elem>(spec: &SplitMergeSpec) -> Program {
    Kernel::record("split_merge", |k| merge_body::<T, U>(k, spec))
}

fn merge_body<'k, T: Elem, U: Elem>(k: &mut Kernel<'k>, spec: &SplitMergeSpec) {
    let SplitMergeSpec { m, n, split, epilogue: epi, tile: [bm, bn], parts } = *spec;
    assert!(!epi.gated && epi.rope.is_none(), "a gated or turned output is not split");
    let elems = parts.map_or(m * n, |p| p.run.count() * p.tile[0] * p.tile[1]);
    let partial = k.input::<F32>("partial", split * elems);
    let bias = epi.bias.then(|| k.input::<T>("bias", n));
    let residual = epi.residual.then(|| k.input::<U>("residual", m * n));
    let c = k.output::<U>("c", m * n);
    let shape = Shape::new(bm, bn);
    let (sum, tile) = match parts {
        None => {
            k.grid([Sc::from(m.div_ceil(bm)), Sc::from(n.div_ceil(bn)), Sc::from(1)]);
            k.warps(4);
            let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
            let bounds = [bound(m, bm), bound(n, bn)];
            let sum = summed(k, split, shape, |k, z| {
                let part = k.view(partial, z * (m * n), [n, 1], shape, bounds.clone());
                k.at(part, row0.clone(), col0.clone())
            });
            (sum, Tile2 { m, n, bm, bn, row0, col0, bounds })
        }
        Some(Parts { order, tile: [tm, tn], run }) => {
            assert!(
                tm.is_multiple_of(bm) && tn.is_multiple_of(bn),
                "a merge tile {:?} divides the GEMM's {:?}",
                [bm, bn],
                [tm, tn]
            );
            // Grid x: the run's tiles; grid y: a tile's merge tiles, row-major.
            let across = tn / bn;
            k.grid([Sc::from(run.count()), Sc::from(tm / bm * across), Sc::from(1)]);
            k.warps(4);
            let (local, sub) = (k.block(0), k.block(1));
            let (tr, tc) = order.walk(local.clone() + run.first(), Sc::from(m.div_ceil(tm)), n.div_ceil(tn));
            let (sr, sc) = (sub.clone() / across, sub % across);
            let (row0, col0) = (tr * tm + sr.clone() * bm, tc * tn + sc.clone() * bn);
            let slot = local * (tm * tn) + sr * (bm * tn) + sc * bn;
            // A merge tile past the edge lies in a GEMM tile across it: cut
            // where those are, in the slots as in the output.
            let tile = Tile2 { m, n, bm, bn, row0, col0, bounds: [bound(m, tm), bound(n, tn)] };
            let within = tile.within();
            let sum = summed(k, split, shape, |k, z| {
                k.view(partial, slot.clone() + z * (run.count() * tm * tn), [tn, 1], shape, within.clone())
            });
            (sum, tile)
        }
    };
    finish::<T, U>(k, vec![sum], epi, bias, residual, None, Dest::Tile(c), &None, &tile);
}

/// The sum of the `split` f32 partials `part` views at each split.
fn summed<'k>(
    k: &mut Kernel<'k>,
    split: usize,
    shape: Shape,
    part: impl Fn(&mut Kernel<'k>, Sc<'k>) -> Gmem<'k, F32>,
) -> Regs<'k, F32> {
    let zero = k.zeros::<F32>(shape);
    let [sum] = k.loop_(split, [zero], |k, z, [acc]| {
        let part = part(k, z);
        let part = k.load(part);
        [k.binary(*acc, part, BinaryOp::Add).into()]
    });
    *sum
}

/// The K-streaming pipeline: the `sa`-shaped A tile `a(step, step·bk)` and
/// `N` B tiles per step into a `stages`-deep shared ring, each operand of a
/// slot held as `slices` k-slices ([`GemmCfg::slices`]), one accumulator
/// per B ([`MmaIn::Acc`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn mainloop<'k, T: MmaIn, const N: usize>(
    k: &mut Kernel<'k>,
    sa: Shape,
    a: &impl Fn(&mut Kernel<'k>, Sc<'k>, Sc<'k>) -> Gmem<'k, T>,
    bs: [Gmem<'k, T>; N],
    trips: usize,
    schedule: Schedule,
    bk: usize,
    first_step: Option<Sc<'k>>,
    slices: usize,
) -> [Acc<'k, T::Acc>; N] {
    let sb = k.shape(bs[0]);
    let (sa_s, sb_s) = (Shape::new(sa.rows, bk / slices), Shape::new(sb.rows, bk / slices));
    let a_s = k.ring::<T>("a_s", sa.elems(), schedule);
    let b_s = [(); N].map(|()| k.ring::<T>("b_s", sb.elems(), schedule));
    let init = [(); N].map(|()| k.zeros::<T::Acc>(Shape::new(sa.rows, sb.rows)));
    k.pipeline(
        trips,
        a_s,
        init,
        |k, step, slot| {
            let step = first_step.map_or(step.clone(), |s| s + step);
            let koff = step.clone() * bk;
            for s in 0..slices {
                let kcol = koff.clone() + s * sa_s.cols;
                let a_g = a(k, step.clone(), kcol.clone());
                let a_g = k.narrow(a_g, sa_s);
                let a_t = k.slot_part(a_s, &slot, s, sa_s);
                k.stage(a_t, a_g, CopyMode::Async);
                for (src, alloc) in bs.into_iter().zip(b_s) {
                    let g = k.at(src, 0, kcol.clone());
                    let g = k.narrow(g, sb_s);
                    let t = k.slot_part(alloc, &slot, s, sb_s);
                    k.stage(t, g, CopyMode::Async);
                }
            }
        },
        |k, _step, slot, mut accs| {
            for s in 0..slices {
                let a_t = k.slot_part(a_s, &slot, s, sa_s);
                let mut i = 0;
                accs = accs.map(|acc| {
                    let b_t = k.slot_part(b_s[i], &slot, s, sb_s);
                    i += 1;
                    k.mma(acc, a_t, false, b_t, true)
                });
            }
            accs
        },
    )
}
