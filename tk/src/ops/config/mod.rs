//! Tile config candidates per target and shape, most promising first. The
//! device picks: the tune store times the first [`SHORTLIST`] tiles that
//! compile clean, most of the lattice, and the ranking orders what is
//! compiled and timed first and stands in for a measurement where there is
//! no device (the first config runs untuned). GEMM-shaped kernels draw
//! theirs from a lattice over the construct's knobs, kept to what the
//! lowering runs and ranked in nanoseconds of SM time from the device's
//! measured [`Costs`]: a round of resident blocks is bound by the SM's issue
//! work or by one wave's own issue, plus the memory latency one pipeline
//! stage ahead cannot hide ([`Planner::pipelined`]). The model orders block
//! tiles well enough to lead with them ([`spread`]) and warp grids less well
//! (on sm_86 the measured best sat as deep as the 92nd of 109 tiles before
//! a trip's pipes overlapped across a SIMD's waves), which is why the
//! shortlist runs that deep. Registers admit nothing here: the
//! layout count over-counts, so the compiled kernel judges ([`shortlist`]).
//! A [`Planner`] holds the rankings of one device. The other kernels' lists
//! are small functions of the target.
//!
//! By responsibility: `cost` the shared pricing core (a [`Run`] of a config,
//! the [`Planner`] that pipelines it), `facts` the device's layer-0
//! description ([`Facts`]), `search` the machinery every family's ranking
//! shares (bounds, the shortlist search, tile twins), and `gemm`,
//! `attention`, `nearest`, `conv`, `norm` their families' configs, lattices
//! and admission.

mod attention;
mod conv;
mod cost;
mod facts;
mod gemm;
mod nearest;
mod norm;
mod search;

pub use self::attention::AttentionProblem;
pub use self::cost::Planner;
pub use self::facts::Facts;
pub use self::gemm::{GemmProblem, gemm_lattice};
pub use self::nearest::NearestProblem;
pub use self::norm::norm_candidates;
pub use self::search::{Rejected, has_kernels};

pub(crate) use self::facts::facts_of;

pub(crate) use self::gemm::gemm_cfg;
pub(crate) use self::nearest::{GROUPS_PAIR, NEAREST_PAIR, Weights, pair_op};
pub(crate) use self::norm::heads_candidates;
pub(crate) use self::search::{
    Compiles, FIRST, SHORTLIST, TWIN_TILES, Tiled, compiled, distinct, distinct_configs, first_clean, representatives,
    shortlist,
};

/// The bounds and cost items only the crate's own tests reach from here;
/// the code that uses them in the library sits in the files they define.
#[cfg(test)]
pub(crate) use self::attention::DIAGONAL;
#[cfg(test)]
pub(crate) use self::cost::Run;
#[cfg(test)]
pub(crate) use self::nearest::REGISTER_K;
#[cfg(test)]
pub(crate) use self::search::{KEPT_TILES, PRUNED_PAST, prune, spread};
