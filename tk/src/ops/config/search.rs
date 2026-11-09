//! The search machinery every family's ranking shares: the bounds that keep
//! a ranking finite, the tile twins a shortlist holds one pipeline of, and
//! the compile-driven search of the first clean configs.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use svod_dtype::GpuArch;

use crate::atoms::Core;
use crate::atoms::Target;
use crate::ir::Schedule;
use crate::kernels::attention::{FaCfg, Forms, WARP_ROWS, narrowest_warp_rows};
use crate::kernels::conv::ConvCfg;
use crate::kernels::gemm::{GemmCfg, TailSplit, TileOrder};
use crate::kernels::nearest::{LaneQueue, NearestCfg, SelectCfg};
use crate::kernels::rows::NormCfg;
use crate::tune::Compiled;

use super::cost::BAND;

/// Targets with kernels: CUDA sm_80+ on `mma.sync` + `cp.async` + `ldmatrix`,
/// sm_90 adding the `wgmma` warpgroup core (no TMA or mbarrier), and AMD
/// matrix cores with register-staged fills: RDNA3/3.5/4 WMMA and CDNA3/4
/// MFMA 16×16×16 on wave64. Metal has atoms but no kernels.
pub fn has_kernels(target: &Target) -> bool {
    match target.arch {
        GpuArch::Cuda(c) => c.major >= 8,
        GpuArch::Amd(a) => a.has_matrix_cores(),
        GpuArch::Metal(_) => false,
    }
}

/// Why a ranking leaves a config out: what the target and the problem's
/// static part decide, settled once per
/// [`Family::Static`](super::cost::Family::Static)
/// ([`Planner::feasible`](super::cost::Planner::feasible)), or what its
/// shape does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    Target(TargetRejected),
    Shape(ShapeRejected),
}

impl From<TargetRejected> for Rejected {
    fn from(why: TargetRejected) -> Self {
        Self::Target(why)
    }
}

impl From<ShapeRejected> for Rejected {
    fn from(why: ShapeRejected) -> Self {
        Self::Shape(why)
    }
}

/// Why a config runs no problem of its statics on the target, whatever the
/// shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetRejected {
    /// It does not lower on the target: no matrix core, a warp grid that
    /// does not tile the core's atoms, a fill that does not split into whole
    /// 16-byte chunks per thread ([`fills_in_chunks`]), a ring depth the
    /// fill does not run ([`Fill::runs`]), lists or key warps the kernel
    /// refuses ([`NearestCfg::lists`], [`FaCfg::shares_keys`]), or lane
    /// queues past the search's bounds.
    Lowers,
    /// Its shared memory passes the block's.
    Smem { need: usize, have: usize },
    /// Its tiles have no register layout.
    Layout,
    /// The registers a lane holds at once pass what it may.
    Registers { need: u32, have: u32 },
    /// It keeps more least scores a row than a register list holds.
    TopK { k: usize, most: usize },
}

/// Why a feasible config is left out of one problem's ranking: its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeRejected {
    /// Its step is shallower than two matrix-core depths where the
    /// reduction allows one.
    Shallow,
    /// Its step does not divide the reduction (a nearest chunk: is deeper
    /// than the whole reduction, a product over zeros; a convolution's: a
    /// tap).
    Divides,
    /// Its tiles pad the output past what a convolution bears.
    Padding,
    /// Its tail split does not run the problem: a batched, gated or turned
    /// GEMM (or a convolution), a tile the merge's does not divide, a split
    /// that does not divide the trips, or whole waves that leave no tail.
    Tail,
}

/// A ranking: its configs with their costs ([`Planner::price`] ns of SM
/// time) and registers by the layout count, cheapest first (then fewer
/// registers) and [pruned](prune), logged under [`crate::ops::PLAN`] with how
/// many of the `spanned` configs it left out.
pub(super) fn sorted<C: Tiled + Debug>(
    problem: &impl Debug,
    mut ranked: Vec<(f64, u32, C)>,
    spanned: usize,
) -> Vec<(f64, u32, C)> {
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let admitted = ranked.len();
    prune(&mut ranked);
    let (pruned, rejected) = (admitted - ranked.len(), spanned - admitted);
    tracing::debug!(target: crate::ops::PLAN, ?problem, admitted, rejected, pruned, "ranking");
    for (rank, (ns, regs, cfg)) in ranked.iter().take(LOGGED).enumerate() {
        tracing::debug!(target: crate::ops::PLAN, rank, ns, regs, ?cfg, "ranked");
    }
    // The shortlist times the first [`SHORTLIST`] tiles of this list and the
    // twins stage the first [`TWIN_TILES`] of its contenders, so a candidate
    // on a tile past them is never measured; a GEMM shortlist walks the
    // [`spread`] order, this the ranked one.
    if tracing::enabled!(target: crate::ops::PLAN, tracing::Level::DEBUG)
        && let Some(&(kept, ..)) = ranked.first()
    {
        let mut seen = HashSet::new();
        let (mut twins, mut shortlist) = (Vec::new(), Vec::new());
        for &(ns, _, c) in &ranked {
            if seen.insert(c.tile()) {
                if seen.len() > TWIN_TILES {
                    twins.push(ns);
                }
                if seen.len() > SHORTLIST {
                    shortlist.push(ns);
                }
            }
        }
        fired("TWIN_TILES", &twins, kept);
        fired("SHORTLIST", &shortlist, kept);
    }
    ranked
}

/// A search bound's fire-log: `bound` cut the modeled costs `dropped` from
/// a ranking whose kept best costs `kept`, and the cheapest of them within
/// [`BAND`] of it is a candidate it may have taken a contender from. One
/// line per shape names the bound, how many it cut within the band and that
/// one's ratio to the kept best; a bound whose cut sat outside the band
/// stays silent. The gathering at each bound runs only with the plan log's
/// debug level on ([`crate::ops::PLAN`]): never by default.
pub(super) fn fired(bound: &str, dropped: &[f64], kept: f64) {
    let within: Vec<f64> = dropped.iter().copied().filter(|cost| *cost <= BAND * kept).collect();
    let Some(best) = within.iter().copied().reduce(f64::min) else { return };
    tracing::debug!(
        target: crate::ops::PLAN,
        "bound {bound} dropped {} within band (best dropped {:.2}x of kept)",
        within.len(),
        best / kept
    );
}

/// Drops the configs of `ranked` (cheapest first) past [`PRUNED_PAST`] times
/// the cheapest one's cost once [`KEPT_TILES`] tiles are kept.
pub(crate) fn prune<C: Tiled>(ranked: &mut Vec<(f64, u32, C)>) {
    let Some(&(cheapest, ..)) = ranked.first() else { return };
    let gather = tracing::enabled!(target: crate::ops::PLAN, tracing::Level::DEBUG);
    let mut tiles = HashSet::new();
    let (mut past, mut kept) = (Vec::new(), Vec::new());
    let mut out = Vec::with_capacity(ranked.len());
    for (cost, regs, c) in ranked.drain(..) {
        let keep = cost <= PRUNED_PAST * cheapest || tiles.len() < KEPT_TILES;
        if keep {
            tiles.insert(c.tile());
            out.push((cost, regs, c));
        } else if gather {
            // Past the cost band [`PRUNED_PAST`] drops it; within, the kept
            // tiles ran out: [`KEPT_TILES`].
            (if cost > PRUNED_PAST * cheapest { &mut past } else { &mut kept }).push(cost);
        }
    }
    *ranked = out;
    if gather {
        fired("PRUNED_PAST", &past, cheapest);
        fired("KEPT_TILES", &kept, cheapest);
    }
}

/// Search bound: the configs of a ranking logged with their costs.
const LOGGED: usize = 16;
/// Search bound: a config this factor past the cheapest one's cost cannot
/// win. Over the shortlists sm_86 timed (GEMMs from 1024³ to 8192²×768,
/// attention, nearest from 8192² to 16384² at depths 128–1024) the measured
/// best was predicted within 1.19 of the cheapest, every config within the
/// tuner's contender band ([`crate::tune::CONTENDER`]) within 1.63. On
/// gfx1201 and gfx1151 (nearest at 16384², depths 256–1024, a 4096³ GEMM,
/// attention) the best was predicted within 1.09, though the model errs
/// wider there: a pruned config ran 1.23 times the best on gfx1201 and 1.08
/// on gfx1151, never within the tuner's drift. Nearest's small candidate
/// blocks run 3.5 to 6 times its best, seconds a launch at 65536² rows.
pub(crate) const PRUNED_PAST: f64 = 2.5;
/// Search bound: the tiles a ranking keeps whatever their cost, so a shortlist
/// whose cheapest tiles fail to compile clean still holds some to time.
pub(crate) const KEPT_TILES: usize = 16;

/// `ranked` (cheapest first, with its costs) led by the band's classes in
/// turn: each class's cheapest tile (classes in the order of their
/// cheapest), then each one's next, and so on, a tile's pipelines (ring
/// depth, unrolling) after it; then the rest as ranked. The model orders
/// the tiles of a class (a block tile's warp grids and depths) better than
/// it orders classes, so a shortlist from the front times every class the
/// band holds rather than one class's variants.
pub(crate) fn spread<C: Tiled, K: Eq + Hash>(ranked: Vec<(f64, C)>, class: impl Fn(&C) -> K) -> Vec<C> {
    let Some(&(best, _)) = ranked.first() else { return vec![] };
    let mut classes: Vec<(K, Vec<C>)> = vec![];
    let mut tiles = HashSet::new();
    for &(_, c) in ranked.iter().take_while(|(cost, _)| *cost <= BAND * best) {
        if !tiles.insert(c.tile()) {
            continue;
        }
        let key = class(&c);
        match classes.iter_mut().find(|(k, _)| *k == key) {
            Some((_, list)) => list.push(c),
            None => classes.push((key, vec![c])),
        }
    }
    let depth = classes.iter().map(|(_, list)| list.len()).max().unwrap_or(0);
    let mut out: Vec<C> = Vec::with_capacity(ranked.len());
    let mut placed = HashSet::new();
    for round in 0..depth {
        for (_, list) in &classes {
            let Some(&lead) = list.get(round) else { continue };
            // The tile's pipelines follow it, in ranked order.
            for &(_, c) in &ranked {
                if c.tile() == lead.tile() && placed.insert(c) {
                    out.push(c);
                }
            }
        }
    }
    out.extend(ranked.into_iter().map(|(_, c)| c).filter(|c| placed.insert(*c)));
    out
}

/// Search bound: the tiles the tune store measures per shape, their
/// pipelines compiled in parallel ([`Search`]): timing a tile costs about
/// 20 ms once the device's clock is up and the ranking cannot be trusted
/// to put the best within a few (see the module docs). 56, measured down
/// from 64 (2026-10-27) by the winners the format-6 stores name: over
/// every replayed choice line of the three cards' stores (162 lines — the
/// phase-E fresh nearest stores, the local sm_86 store and the
/// device-suite stores' gemm, attention and conv lines), the share whose
/// measured winner's tile sits in the model's top N. At 56 that is 161 of
/// 162 (99.4 %): every family and card whole — nearest (the expensive
/// family, 16384² rows at depths 128–768) from 24 tiles on sm_86, gemm at
/// 32, attention and conv at 16 — but one int8 line of gfx1151's six,
/// whose winner sat at the 62nd tile. 48, tried first, holds only 95 %
/// (five conv lines and a nearest line of sm_86's out at 49–54) and the
/// battery refused it: the gelu GEMM's winner (704×2048×512, its tile at
/// 54 — a warp grid the cheaper [2,2] form of the same tile hides from
/// the tile count) fell out of the shortlist and the row read 4–5 % over
/// four alternating tunes; at 56 the winner is timed again and picked by
/// both (70.4–70.8 µs the base, 71.7–72.5 the tip, one tile — the runs'
/// own noise). The rest of the battery at 48 was already same-or-better
/// than 64 (every nearest, gemm, attention and rerank row on sm_86 and
/// gfx1201, the conv classes within +1.9 %), and 56 sits between the two.
/// The tuning wall-clock is unmoved on sm_86 at any of 48, 56 and 64
/// (17.5–20.5 s at 16384² d128 k64 — the measurement discipline, the
/// clock warm-up and the runs to a stable minimum, dominates the count)
/// and 10–20 % less on gfx1201 (8.9–9.6 s at 48 against 11.5–11.6 at 64,
/// whose clang compiles the shorter list does not pay), with 25–12 %
/// fewer compiles in flight everywhere.
pub(crate) const SHORTLIST: usize = 56;
/// Search bound: the most tiles of a timing whose other pipelines are
/// timed after it, among those within the tuner's contender band
/// ([`crate::tune::CONTENDER`]) of the leader: a tile enters the shortlist
/// as its first clean pipeline, on gfx1201 the 4096³ GEMM's measured best
/// entered 8 % behind its best and its standing drifts between fourth and
/// tenth across runs while the top ten sit within 10 %, so a fixed depth
/// cut it on some runs. Over three cards' exhaustive sweeps eight reaches
/// every tile a deeper stage would.
pub(crate) const TWIN_TILES: usize = 8;

/// A config's tile: all of it but its pipeline (ring depth, unrolling),
/// which the ranking orders by a cost model alone and the compiler may
/// refuse, so the shortlist holds one config per tile, its first clean
/// pipeline ([`shortlist`]); the tile that wins has its other pipelines,
/// its [`twins`](Self::twins), timed after.
pub(crate) trait Tiled: Copy + Eq + Hash + Send + Field {
    fn tile(&self) -> Self;

    /// The configs of `ranked` sharing this tile, each under this config's
    /// variant (a split count the tuner added): every pipeline of the tile.
    fn twins(&self, ranked: &[Self]) -> Vec<Self>;

    /// Whether the config may stand for its tile in the shortlist and the
    /// first clean search; one that may not (a variant the twins stage
    /// explores) is timed only as a twin of a contender tile.
    fn represents(&self) -> bool {
        true
    }

    /// The config as the compact code a store line carries beside its
    /// index ([`super::super::tune`]'s format): its [`Field`]s in order
    /// between commas, so a line's candidates read without re-deriving the
    /// list their digest names.
    fn code(&self) -> String {
        let mut out = vec![];
        self.put(&mut out);
        out.join(",")
    }

    /// The config a [`code`](Self::code) holds; `None` where it is not the
    /// family's form.
    fn of_code(code: &str) -> Option<Self> {
        let mut fields = code.split(',').peekable();
        let cfg = Self::take(&mut fields)?;
        fields.next().is_none().then_some(cfg)
    }
}

/// A code's fields between its commas.
type Fields<'a> = std::iter::Peekable<std::str::Split<'a, char>>;

/// A field of a config's code: the tokens it writes, and what it reads back
/// from them (`None` where they are not its form).
pub(crate) trait Field: Sized {
    fn put(&self, out: &mut Vec<String>);
    fn take(fields: &mut Fields<'_>) -> Option<Self>;
}

macro_rules! number {
    ($($ty:ty),*) => {$(
        impl Field for $ty {
            fn put(&self, out: &mut Vec<String>) {
                out.push(self.to_string());
            }
            fn take(fields: &mut Fields<'_>) -> Option<Self> {
                fields.next()?.parse().ok()
            }
        }
    )*};
}
number!(usize, u32);

impl Field for bool {
    fn put(&self, out: &mut Vec<String>) {
        out.push(u8::from(*self).to_string());
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        match fields.next()? {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        }
    }
}

/// `~` a `None`, else the value's own fields.
impl<T: Field> Field for Option<T> {
    fn put(&self, out: &mut Vec<String>) {
        match self {
            Some(v) => v.put(out),
            None => out.push("~".into()),
        }
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        if fields.peek() == Some(&"~") {
            fields.next();
            return Some(None);
        }
        T::take(fields).map(Some)
    }
}

impl<T: Field, const N: usize> Field for [T; N] {
    fn put(&self, out: &mut Vec<String>) {
        self.iter().for_each(|v| v.put(out));
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        let mut out = Vec::with_capacity(N);
        for _ in 0..N {
            out.push(T::take(fields)?);
        }
        out.try_into().ok()
    }
}

/// `w`, `x` or `g`.
impl Field for Core {
    fn put(&self, out: &mut Vec<String>) {
        out.push(
            match self {
                Core::Warp => "w",
                Core::Wide => "x",
                Core::Group => "g",
            }
            .into(),
        );
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        match fields.next()? {
            "w" => Some(Core::Warp),
            "x" => Some(Core::Wide),
            "g" => Some(Core::Group),
            _ => None,
        }
    }
}

/// A ring `r<depth>`, `u` after it when unrolled; two crews `p<kslice>`.
impl Field for Schedule {
    fn put(&self, out: &mut Vec<String>) {
        out.push(match self {
            Schedule::Ring { depth, unroll } => format!("r{depth}{}", if *unroll { "u" } else { "" }),
            Schedule::PingPong { kslice } => format!("p{kslice}"),
        });
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        let field = fields.next()?;
        let (kind, rest) = field.split_at(field.len().min(1));
        match kind {
            "r" => {
                let (depth, unroll) = rest.strip_suffix('u').map_or((rest, false), |depth| (depth, true));
                Some(Schedule::Ring { depth: depth.parse().ok()?, unroll })
            }
            "p" => Some(Schedule::PingPong { kslice: rest.parse().ok()? }),
            _ => None,
        }
    }
}

/// `g` and the group's tile rows.
impl Field for TileOrder {
    fn put(&self, out: &mut Vec<String>) {
        out.push(format!("g{}", self.rows()));
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        TileOrder::grouped(fields.next()?.strip_prefix('g')?.parse().ok()?)
    }
}

/// `t` and the blocks a tail tile splits over.
impl Field for TailSplit {
    fn put(&self, out: &mut Vec<String>) {
        out.push(format!("t{}", self.split()));
    }
    fn take(fields: &mut Fields<'_>) -> Option<Self> {
        TailSplit::new(fields.next()?.strip_prefix('t')?.parse().ok()?)
    }
}

/// A config's code: its fields in order, each a [`Field`]; the comment
/// gives a code.
macro_rules! coded {
    ($(#[$doc:meta])* $ty:ident { $($field:ident),* }) => {
        $(#[$doc])*
        impl Field for $ty {
            fn put(&self, out: &mut Vec<String>) {
                $(self.$field.put(out);)*
            }
            fn take(fields: &mut Fields<'_>) -> Option<Self> {
                Some(Self { $($field: Field::take(fields)?),* })
            }
        }
    };
}
coded!(SelectCfg { warps, queue, filter });
coded!(LaneQueue { width, depth, passes });
coded!(Forms { truncate_p, sum_after, skip_rescale, raw_max, bias_direct });
coded!(
    /// `w,128,256,64,2,4,r3u,g8,~` (`t4` for a tail split four ways)
    GemmCfg { core, tile, warps, schedule, order, tail }
);
coded!(
    /// `w,64,64,32,2,2,r2,g8,~,4`
    ConvCfg { gemm, split }
);
coded!(
    /// `64,64,r3,1,1,0,0,1,1,1,16,g`
    FaCfg { bq, bkv, schedule, splits, forms, key_warps, warp_rows, core }
);
coded!(
    /// `4`
    NormCfg { br }
);
coded!(
    /// `64,64,r2,16,~,1,~,~,4,32,0`
    NearestCfg { bq, bkv, schedule, warp_rows, bk, splits, queue, select, lanes }
);

/// The configs of `ranked` that may stand for their tiles, in order.
pub(crate) fn representatives<C: Tiled>(ranked: &[C]) -> Vec<C> {
    ranked.iter().copied().filter(Tiled::represents).collect()
}

/// A tile's raster and tail split are variants of it, as its pipelines
/// are: the planner walks each tile at the raster and split it prices
/// cheapest and offers one alternative beside it where it is unsure
/// ([`GemmProblem::ranked`](super::GemmProblem)), timed only as a twin of a
/// contender tile.
impl Tiled for GemmCfg {
    fn tile(&self) -> Self {
        let tile = [self.tile[0], self.tile[1], 0];
        GemmCfg { tile, schedule: self.schedule.tile(), order: TileOrder::DEFAULT, tail: None, ..*self }
    }

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        ranked.iter().filter(|c| c.tile() == self.tile()).copied().collect()
    }
}

impl Tiled for ConvCfg {
    fn tile(&self) -> Self {
        ConvCfg { gemm: self.gemm.tile(), ..*self }
    }

    /// A split's twins keep its depth as well as its split: a partial
    /// kernel's split must divide its trips, and a tile's depths divide
    /// differently (3456 taps are 108 trips at a depth of 32, 54 at 64).
    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        self.gemm
            .twins(&ranked.iter().map(|c| c.gemm).collect::<Vec<_>>())
            .into_iter()
            .filter(|gemm| self.split == 1 || gemm.tile[2] == self.gemm.tile[2])
            .map(|gemm| ConvCfg { gemm, ..*self })
            .collect()
    }
}

/// The blocks, the splits and the softmax forms: a forms pair is two tiles.
/// A tile is its query and key block sizes and its key warps: a different
/// number of key warps is a different kernel structure, not a pipeline of
/// the same one. Its rows per warp are a variant its twins carry, as its
/// pipelines are: the one-row config represents the tile.
impl Tiled for FaCfg {
    fn tile(&self) -> Self {
        FaCfg { schedule: self.schedule.tile(), warp_rows: WARP_ROWS, ..*self }
    }

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        let block = |c: &FaCfg| (c.bq, c.bkv, c.key_warps, c.core);
        ranked
            .iter()
            .filter(|c| block(c) == block(self))
            .map(|c| FaCfg { schedule: c.schedule, warp_rows: c.warp_rows, ..*self })
            .collect()
    }

    /// Two atom rows per warp are a variant of a tile that holds one; the
    /// wide core offers one rows per warp a block.
    fn represents(&self) -> bool {
        self.core == Core::Wide || self.warp_rows == narrowest_warp_rows(self.core, WARP_ROWS, self.bq)
    }
}

impl Tiled for NormCfg {
    fn tile(&self) -> Self {
        *self
    }

    fn twins(&self, _: &[Self]) -> Vec<Self> {
        vec![*self]
    }
}

/// A tile is its query and candidate blocks, its rows per warp, its depth
/// chunk and its lists; its ring depth and unrolling are its pipelines, and
/// whether its lane queues take a trip in rounds or in passes its twins too
/// (timed where the tile leads, never taking a place of the shortlist).
impl Tiled for NearestCfg {
    fn tile(&self) -> Self {
        let lanes = self.lanes.map(|l| LaneQueue { passes: false, ..l });
        NearestCfg { schedule: self.schedule.tile(), lanes, ..*self }
    }

    /// The ranked (unsplit) pipelines of the tile, at the split the tuner
    /// gave this config.
    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        let split = |c: &Self| NearestCfg { splits: self.splits, ..*c };
        ranked.iter().map(split).filter(|c| c.tile() == self.tile()).collect()
    }
}

/// The first config of each tile of `ranked`, in ranked order.
pub(crate) fn distinct<C: Tiled>(ranked: &[C]) -> impl Iterator<Item = C> + '_ {
    let mut seen = HashSet::new();
    ranked.iter().copied().filter(move |c| seen.insert(c.tile()))
}

pub(crate) fn distinct_configs<C: Tiled>(list: &[C]) -> Vec<C> {
    let mut seen = HashSet::new();
    list.iter().copied().filter(|c| seen.insert(*c)).collect()
}

/// The indices of the first [`SHORTLIST`] configs of `ranked` whose
/// compiled kernels do not spill, one per tile (its first clean pipeline,
/// so a spilling unrolled ring yields to its rolled twin), in ranked order;
/// after them, within the [`SHORTLIST`], the tiles none of whose pipelines
/// compiled clean, each by its first spilling one: the compiled kernel
/// judges what it can and the time the rest (a spilling kernel still beats
/// the graph's, and on gfx1201 the 16384² depth-768 nearest's best tile,
/// 128 rows by 128 candidates at 32 rows a warp, spills 196 bytes a lane
/// and runs 86.4 Gpairs/s to the clean pick's 84.7).
/// `compiles` is handed what the [`Search`] wants as soon as it wants it
/// (the tuner compiles in parallel, in the order handed), each verdict
/// taken as it lands, until nothing more is wanted. `None` when nothing
/// compiled: no device to judge on.
pub(crate) fn shortlist<C: Tiled>(ranked: &[C], compiles: &mut impl Compiles<C>) -> Option<Vec<usize>> {
    let mut search = Search::new(ranked, SHORTLIST);
    loop {
        while let Some(at) = search.next() {
            compiles.start(at, ranked[at]);
        }
        while compiles.idle()
            && let Some(at) = search.spare()
        {
            compiles.start(at, ranked[at]);
        }
        let Some((at, verdict)) = compiles.finished() else { return search.judged() };
        search.record(at, verdict);
    }
}

/// Compiles started by a ranked index, their verdicts back in any order.
pub(crate) trait Compiles<C> {
    fn start(&mut self, at: usize, config: C);
    /// Whether a compile started now would run at once, beside nothing queued.
    fn idle(&self) -> bool;
    /// A compile that finished, waiting for one; `None` when none is in flight.
    fn finished(&mut self) -> Option<(usize, Compiled)>;
}

/// The shortlist's search of a ranked list: it compiles what a walk in
/// ranked order compiling one config at a time compiles, so the verdicts
/// alone decide, never the order they come back in. That walk stops at the
/// `limit`th tile found clean and skips a config once a pipeline of its
/// tile ahead of it is clean; here a config is wanted ([`Self::next`]) once
/// every pipeline of its tile ahead of it is judged unclean and fewer than
/// `limit` tiles ahead of it are clean or have a compile in flight. Where
/// the compiler has room, the next pipeline of a tile whose pipeline ahead
/// is still compiling may start early ([`Self::spare`]): a slow compile
/// (one killed at its deadline) holds no chain of its tile's pipelines up
/// one after another, and the walk ignores what it would not have asked.
struct Search {
    /// Each config's tile, numbered in order of first appearance.
    tiles: Vec<usize>,
    limit: usize,
    verdicts: Vec<Option<Compiled>>,
    asked: Vec<bool>,
}

impl Search {
    fn new<C: Tiled>(ranked: &[C], limit: usize) -> Self {
        let mut ids = HashMap::new();
        let tiles = ranked
            .iter()
            .map(|c| {
                let next = ids.len();
                *ids.entry(c.tile()).or_insert(next)
            })
            .collect();
        Self { tiles, limit, verdicts: vec![None; ranked.len()], asked: vec![false; ranked.len()] }
    }

    /// The next config wanted, marked asked; `None` while the rest wait on
    /// verdicts in flight, and once nothing more is wanted.
    fn next(&mut self) -> Option<usize> {
        let (mut clean, mut open) = (HashSet::new(), HashSet::new());
        for (at, &tile) in self.tiles.iter().enumerate() {
            if clean.len() + open.len() >= self.limit {
                return None;
            }
            if clean.contains(&tile) || open.contains(&tile) {
                continue;
            }
            match self.verdicts[at] {
                Some(Compiled::Clean) => _ = clean.insert(tile),
                Some(_) => {}
                None if self.asked[at] => _ = open.insert(tile),
                None => {
                    self.asked[at] = true;
                    return Some(at);
                }
            }
        }
        None
    }

    /// A pipeline the walk may yet want, marked asked: the first not asked
    /// of a tile whose pipeline ahead of it is compiling, within the tiles
    /// [`Self::next`] reaches. Its verdict counts only where the walk asks
    /// for it: a clean pipeline ahead of it takes its tile first, and past
    /// the `limit`th clean tile the shortlist ends.
    fn spare(&mut self) -> Option<usize> {
        let (mut clean, mut open) = (HashSet::new(), HashSet::new());
        for (at, &tile) in self.tiles.iter().enumerate() {
            if clean.len() + open.len() >= self.limit {
                return None;
            }
            if clean.contains(&tile) {
                continue;
            }
            if open.contains(&tile) {
                if self.verdicts[at].is_none() && !self.asked[at] {
                    self.asked[at] = true;
                    return Some(at);
                }
                continue;
            }
            match self.verdicts[at] {
                Some(Compiled::Clean) => _ = clean.insert(tile),
                None if self.asked[at] => _ = open.insert(tile),
                _ => {}
            }
        }
        None
    }

    fn record(&mut self, at: usize, verdict: Compiled) {
        self.verdicts[at] = Some(verdict);
    }

    /// The shortlist the verdicts make ([`shortlist`]): the clean tiles,
    /// then the spilling ones, a pipeline each.
    fn judged(&self) -> Option<Vec<usize>> {
        let mut tiles = HashSet::new();
        let at = |want| self.verdicts.iter().enumerate().filter(move |(_, v)| **v == Some(want)).map(|(i, _)| i);
        let first = |i: &usize| tiles.insert(self.tiles[*i]);
        let out: Vec<usize> = at(Compiled::Clean).chain(at(Compiled::Spilled)).filter(first).take(self.limit).collect();
        (!out.is_empty()).then_some(out)
    }
}

/// The indices of every config of `list` that compiles, spilling or not,
/// in order; `None` when nothing does. A tile's [twins](Tiled::twins) are
/// judged so: the shortlist set its spilling pipelines aside for a clean
/// one, but among a tile's pipelines the time alone decides, and an
/// unrolled ring that spills a little may be the fastest.
pub(crate) fn compiled<C: Tiled>(list: &[C], mut compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<Vec<usize>> {
    let mut out = vec![];
    for (n, chunk) in list.chunks(SHORTLIST).enumerate() {
        let built = compile(chunk).into_iter().enumerate().filter(|(_, c)| *c != Compiled::Failed);
        out.extend(built.map(|(i, _)| n * SHORTLIST + i));
    }
    (!out.is_empty()).then_some(out)
}

/// Search bound: configs compiled at once in search of the first clean one: the first
/// ranked config nearly always is, and a search is what runs with tuning
/// off, where no test should pay for a lattice.
pub(crate) const FIRST: usize = 8;

/// The index of the first config of `ranked` whose compiled kernel does
/// not spill, judged as [`shortlist`] judges (every one spilling: the first
/// that compiles; nothing compiling: `None`), `compile` asked about batches
/// of [`FIRST`] in ranked order until one passes.
pub(crate) fn first_clean<C: Tiled>(ranked: &[C], mut compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<usize> {
    let mut spilled = None;
    for (n, chunk) in ranked.chunks(FIRST).enumerate() {
        for (i, compiled) in compile(chunk).into_iter().enumerate() {
            match compiled {
                Compiled::Clean => return Some(n * FIRST + i),
                Compiled::Spilled => _ = spilled.get_or_insert(n * FIRST + i),
                Compiled::Failed => {}
            }
        }
    }
    spilled
}
