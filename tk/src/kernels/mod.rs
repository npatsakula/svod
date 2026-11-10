//! The kernels as tile programs: a spec (shapes, options) admitted with its
//! tile config in, a [`Program`](crate::ir::Program) out, generic over the
//! element type.

pub mod attention;
pub mod conv;
pub mod gemm;
pub mod heads;
pub mod nearest;
pub mod rows;

use std::f64::consts::{FRAC_1_SQRT_2, LOG2_E};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::build::*;
use crate::ir::*;
use crate::ops::config::{Family, Planner, Rejected};

/// A kernel spec and the config the planner admitted for it: the config is
/// feasible on the planner's target for the problem the spec states and
/// claimed on its shape, through the one admission path (each spec's
/// `admitted`, which derives the problem from the spec itself). A kernel
/// takes its spec as one, so a kernel is built from nothing the planner did
/// not let through for that very spec, and the checks the admission makes
/// are not made again where the kernel is built. `'t` is the planner's
/// borrow. Neither half changes after admission: the spec is read through
/// [`Deref`], the config through [`Admitted::cfg`]. A config no planner
/// admitted for the spec is taken only through the `unsafe`
/// [`Admitted::assume`].
///
/// A spec admitted for 4096³ does not run 40×40×72, nor a gated epilogue on
/// a config admitted ungated: an admitted spec is read-only,
///
/// ```compile_fail,E0594
/// # use svod_tk::kernels::{Admitted, gemm::GemmSpec};
/// fn shrink(spec: &mut Admitted<'_, GemmSpec>) {
///     spec.k = 72;
/// }
/// ```
///
/// is minted by admission alone,
///
/// ```compile_fail,E0451
/// # use std::marker::PhantomData;
/// # use svod_tk::kernels::{Admitted, gemm::GemmSpec};
/// fn swap<'t>(admitted: Admitted<'t, GemmSpec>, other: GemmSpec) -> Admitted<'t, GemmSpec> {
///     Admitted { spec: other, cfg: admitted.cfg(), planner: PhantomData }
/// }
/// ```
///
/// and is what a kernel takes:
///
/// ```compile_fail,E0308
/// # use svod_tk::build::BF16;
/// # use svod_tk::kernels::{Batch, gemm::{Epilogue, GemmSpec, gemm}};
/// let spec = GemmSpec { m: 40, n: 40, k: 72, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT };
/// gemm::<BF16>(&spec);
/// ```
///
/// Taking a config as admitted without the planner is `unsafe`:
///
/// ```compile_fail,E0133
/// # use svod_tk::kernels::{Admitted, gemm::{GemmCfg, GemmSpec}};
/// fn forge(spec: GemmSpec, cfg: GemmCfg) -> Admitted<'static, GemmSpec> {
///     Admitted::assume(spec, cfg)
/// }
/// ```
#[derive(Clone, PartialEq)]
pub struct Admitted<'t, S: Spec> {
    spec: S,
    cfg: S::Cfg,
    planner: PhantomData<&'t ()>,
}

/// A kernel's description: its shapes and options, everything but the
/// config the planner admits for it ([`Admitted`]).
pub trait Spec {
    type Cfg: Copy + PartialEq + fmt::Debug;
}

/// A spec the planner admits: the problem it ranks and admits for it.
pub(crate) trait Admits: Spec + Sized {
    type Problem: Family<Cfg = Self::Cfg>;

    fn problem(&self) -> Self::Problem;

    /// `cfg` admitted for this spec on `planner`, or why it is not.
    fn admit(self, planner: &Planner, cfg: Self::Cfg) -> Result<Admitted<'_, Self>, Rejected> {
        planner.claim_of(&self.problem(), &cfg).map(|_| Admitted::mint(self, cfg))
    }

    /// `cfg`, which `planner` ranked for this spec or the tuner derived from
    /// one it ranked, admitted: a refusal is the planner's own inconsistency.
    fn admit_ranked(self, planner: &Planner, cfg: Self::Cfg) -> Admitted<'_, Self>
    where
        Self: fmt::Debug,
    {
        let spec = format!("{self:?}");
        self.admit(planner, cfg).unwrap_or_else(|why| panic!("{cfg:?} ranked for {spec} is refused: {why:?}"))
    }
}

impl<'t, S: Spec> Admitted<'t, S> {
    fn mint(spec: S, cfg: S::Cfg) -> Self {
        Self { spec, cfg, planner: PhantomData }
    }

    /// `spec` with `cfg` taken as admitted: for a kernel built outside the
    /// op layer.
    ///
    /// # Safety
    ///
    /// The kernel built from it checks nothing admission checks: where `cfg`
    /// is not feasible on the launch's target for `spec`'s problem, or its
    /// claim does not fit the spec's shape, the kernel may read or write
    /// past its operands or compute the wrong values.
    pub unsafe fn assume(spec: S, cfg: S::Cfg) -> Self {
        Self::mint(spec, cfg)
    }

    /// The admitted config.
    pub fn cfg(&self) -> S::Cfg {
        self.cfg
    }

    /// The spec, its admission dropped.
    pub fn into_spec(self) -> S {
        self.spec
    }

    /// The spec `f` derives from this one, kept under its config: for the
    /// op layer's own variants of an admitted spec (a measured batch, a
    /// timed key length, a chunk of the rows), which the kernel runs within
    /// what was admitted.
    pub(crate) fn map<U: Spec<Cfg = S::Cfg>>(&self, f: impl FnOnce(S) -> U) -> Admitted<'t, U>
    where
        S: Clone,
    {
        Admitted::mint(f(self.spec.clone()), self.cfg)
    }
}

impl<S: Spec> Deref for Admitted<'_, S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.spec
    }
}

impl<S: Spec + fmt::Debug> fmt::Debug for Admitted<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Admitted").field("spec", &self.spec).field("cfg", &self.cfg).finish()
    }
}

/// The launch's batch axis, grid z.
#[derive(Clone, PartialEq, Eq)]
pub enum Batch {
    /// `n` batches, all launched; a single one is no addressing axis.
    Static(usize),
    /// A runtime variable bound by name; buffers hold its `max`.
    Var(Var),
}

/// The tune store keys by this text: a bound batch prints as it did when
/// the variable's fields were the variant's.
impl fmt::Debug for Batch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Batch::Static(n) => f.debug_tuple("Static").field(n).finish(),
            Batch::Var(v) => v.fmt(f),
        }
    }
}

impl Batch {
    pub fn var(name: impl Into<String>, min: i64, max: i64) -> Self {
        Batch::Var(Var { name: name.into(), min, max })
    }

    pub fn capacity(&self) -> usize {
        match self {
            Batch::Static(n) => *n,
            Batch::Var(v) => v.max as usize,
        }
    }

    /// The grid z extent and the batch index, `None` for a single static batch.
    fn axis<'k>(&self, k: &mut Kernel<'k>) -> (Sc<'k>, Option<Sc<'k>>) {
        match self {
            Batch::Static(1) => (Sc::from(1), None),
            Batch::Static(n) => (Sc::from(*n), Some(k.block(2))),
            Batch::Var(v) => (k.var(v.name.clone(), v.min, v.max), Some(k.block(2))),
        }
    }
}

/// `batch · stride`, the element offset of the current batch.
fn batch_offset<'k>(batch: &Option<Sc<'k>>, stride: usize) -> Sc<'k> {
    batch.clone().map_or(Sc::from(0), |b| b * stride)
}

/// A row/column bound only where the tile grid overhangs `len`.
fn bound<'k>(len: usize, tile: usize) -> Option<Sc<'k>> {
    (!len.is_multiple_of(tile)).then(|| Sc::from(len))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Act {
    #[default]
    None,
    /// `0.5·x·(1 + erf(x/√2))`, erf by Abramowitz–Stegun 7.1.26 (|error| ≤ 1.5e-7).
    Gelu,
    /// `x·sigmoid(x)`.
    Silu,
}

impl Act {
    fn apply<'k>(self, k: &mut Kernel<'k>, x: Regs<'k, F32>) -> Regs<'k, F32> {
        match self {
            Act::None => x,
            Act::Gelu => gelu(k, x),
            Act::Silu => silu(k, x),
        }
    }
}

fn konst<'k>(k: &mut Kernel<'k>, like: Regs<'k, F32>, v: f64) -> Regs<'k, F32> {
    let shape = k.shape(like);
    k.fill(shape, Const::Float(v))
}

fn affine<'k>(k: &mut Kernel<'k>, x: Regs<'k, F32>, mul: f64, add: f64) -> Regs<'k, F32> {
    let m = konst(k, x, mul);
    let x = k.binary(x, m, BinaryOp::Mul);
    let a = konst(k, x, add);
    k.binary(x, a, BinaryOp::Add)
}

/// `x / (1 + 2^(-x·log2 e))`.
fn silu<'k>(k: &mut Kernel<'k>, x: Regs<'k, F32>) -> Regs<'k, F32> {
    let e = affine(k, x, -LOG2_E, 0.0);
    let e = k.unary(e, UnaryOp::Exp2);
    let one = konst(k, x, 1.0);
    let d = k.binary(e, one, BinaryOp::Add);
    k.binary(x, d, BinaryOp::Div)
}

/// `0.5·(x + |x|·erf(|x|/√2))`, the sign of erf folded into `|x|`.
fn gelu<'k>(k: &mut Kernel<'k>, x: Regs<'k, F32>) -> Regs<'k, F32> {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [0.254_829_592, -0.284_496_736, 1.421_413_741, -1.453_152_027, 1.061_405_429];
    let ax = k.unary(x, UnaryOp::Abs);
    let z = affine(k, ax, FRAC_1_SQRT_2, 0.0);
    let t = affine(k, z, P, 1.0);
    let t = k.unary(t, UnaryOp::Recip);
    let mut poly = konst(k, t, A[4]);
    for a in A[..4].iter().rev() {
        let p = k.binary(poly, t, BinaryOp::Mul);
        poly = affine(k, p, 1.0, *a);
    }
    let poly = k.binary(poly, t, BinaryOp::Mul);
    let z2 = k.binary(z, z, BinaryOp::Mul);
    let e = affine(k, z2, -LOG2_E, 0.0);
    let e = k.unary(e, UnaryOp::Exp2);
    let tail = k.binary(poly, e, BinaryOp::Mul);
    let erf = affine(k, tail, -1.0, 1.0);
    let s = k.binary(ax, erf, BinaryOp::Mul);
    let s = k.binary(x, s, BinaryOp::Add);
    affine(k, s, 0.5, 0.0)
}

/// A 16-bit tile loaded and widened to f32.
fn load_f32<'k, T: Elem, P: Readable>(k: &mut Kernel<'k>, view: Tile<'k, P, T>) -> Regs<'k, F32> {
    let v = k.load(view);
    k.cast::<T, F32>(v)
}
