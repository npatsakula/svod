//! Which path an op takes, as pure functions of the operands' dims at
//! capacity, the bound batch variable's presence, the dtypes and the target.

use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::{ConstValue, Op, SInt, ops};
use svod_tensor::Tensor;

use super::config::{self, Planner};
use crate::ir::Var;
use crate::kernels::attention::FaCfg;
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::GemmCfg;
use crate::kernels::nearest::{Floors, MAX_CANDIDATES, Metric, NearestCfg};
use crate::kernels::rows::NormCfg;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan<C> {
    /// The tile configs, ranked by the model: untuned, the first that
    /// compiles without spilling runs; tuned, the shortlist is drawn from the front.
    Kernel(Arc<[C]>),
    Graph(Fallback),
}

/// Why the graph op runs instead of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// No tk target for the device.
    Target,
    /// f32 (precision policy) or another type without a kernel on the target.
    Dtype,
    /// A symbolic dim other than a bound leading one.
    Symbolic,
    /// A shape the kernel does not cover (head dim, row width, reduction dim).
    Shape,
    /// No tile config fits the target.
    Config,
}

/// An operand's dims at capacity and whether dim 0 is a bound variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extent {
    pub dims: Vec<usize>,
    pub var: bool,
}

/// `t`'s dims at capacity and its leading dim where a runtime variable (the
/// JIT `batch_var`) binds it, as the operand spells it; `None` when a dim
/// other than a bound leading one is symbolic.
pub fn extent(shape: &[SInt]) -> Option<(Extent, Option<SInt>)> {
    let var = match shape.first() {
        Some(dim @ SInt::Symbolic(_)) => {
            bound_var(dim)?;
            Some(dim.clone())
        }
        _ => None,
    };
    let dims: Option<Vec<usize>> =
        shape.iter().enumerate().map(|(i, d)| if i == 0 { d.vmax() } else { d.as_const() }).collect();
    Some((Extent { dims: dims?, var: var.is_some() }, var))
}

/// The variable bound to `dim`, as a kernel binds it at launch: `None` for
/// a constant or another expression.
pub fn bound_var(dim: &SInt) -> Option<Var> {
    let SInt::Symbolic(dim) = dim else { return None };
    let var = match dim.op() {
        Op::Bind(ops::Bind { var, .. }) => var,
        _ => dim,
    };
    let int = |v: &ConstValue| match *v {
        ConstValue::Int(v) => Some(v),
        ConstValue::UInt(v) => i64::try_from(v).ok(),
        _ => None,
    };
    match var.op() {
        Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => {
            Some(Var { name: name.clone(), min: *min_val, max: *max_val })
        }
        Op::Param(ops::Param { arg, .. }) => {
            let (min, max) = arg.vmin_vmax.as_ref()?;
            Some(Var { name: arg.name.clone()?, min: int(&min.0)?, max: int(&max.0)? })
        }
        _ => None,
    }
}

pub fn shape_of(t: &Tensor) -> svod_tensor::error::Result<Vec<SInt>> {
    Ok(t.shape()?.to_vec())
}

/// The planner when its target has kernels and every operand shares a 16-bit
/// type it has a matrix core for (f32 keeps the graph by policy).
fn gate<'a>(planner: Option<&'a Planner>, dtypes: &[DType]) -> Result<&'a Planner, Fallback> {
    let planner = planner.filter(|p| config::has_kernels(&p.device.target)).ok_or(Fallback::Target)?;
    let first = dtypes[0].scalar().ok_or(Fallback::Dtype)?;
    let typed = matches!(first, ScalarDType::BFloat16 | ScalarDType::Float16)
        && planner.device.target.mma(first, ScalarDType::Float32).is_some()
        && dtypes.iter().all(|d| d.scalar() == Some(first));
    typed.then_some(planner).ok_or(Fallback::Dtype)
}

/// The last dim, which a bound batch variable must not be.
fn reduced(x: &Extent) -> Result<usize, Fallback> {
    match x.dims[..] {
        [] => Err(Fallback::Shape),
        [_] if x.var => Err(Fallback::Symbolic),
        [.., last] => Ok(last),
    }
}

fn plan<C>(f: impl FnOnce() -> Result<Arc<[C]>, Fallback>) -> Plan<C> {
    f().map_or_else(Plan::Graph, Plan::Kernel)
}

fn candidates<C>(list: Arc<[C]>, none: Fallback) -> Result<Arc<[C]>, Fallback> {
    if list.is_empty() { Err(none) } else { Ok(list) }
}

/// `x [lead..., k] · w [n·halves, k]ᵀ`; `dtypes` are those of every operand.
pub fn linear(planner: Option<&Planner>, dtypes: &[DType], x: Option<&Extent>, n: usize, gated: bool) -> Plan<GemmCfg> {
    plan(|| {
        let (planner, x) = (gate(planner, dtypes)?, x.ok_or(Fallback::Symbolic)?);
        let k = reduced(x)?;
        // A bound batch walks grid z; each batch is a GEMM over the rows behind it.
        let lead = &x.dims[..x.dims.len() - 1];
        let (batches, rows) = if x.var { (lead[0], lead[1..].iter().product()) } else { (1, lead.iter().product()) };
        // Columns are stored in vector runs: a run never straddles `n`.
        if !n.is_multiple_of(8) || rows * batches == 0 {
            return Err(Fallback::Shape);
        }
        candidates(planner.gemm_ranked(batches, rows, n, k, gated), Fallback::Config)
    })
}

/// `x [b, h, w, cin]` ⊛ `geom`'s weight; `dtypes` are those of `x`, `w` and
/// the bias, `out` the output type and `residual` the residual's. A static
/// batch folds into the GEMM rows, a bound one walks grid z.
pub fn conv2d(
    planner: Option<&Planner>,
    dtypes: &[DType],
    out: DType,
    residual: Option<DType>,
    x: Option<&Extent>,
    geom: &ConvGeom,
    groups: usize,
) -> Plan<ConvCfg> {
    plan(|| {
        let (planner, x) = (gate(planner, dtypes)?, x.ok_or(Fallback::Symbolic)?);
        let wide = out == DType::Float32;
        if !(out == dtypes[0] || wide) || residual.is_some_and(|r| r != out) {
            return Err(Fallback::Dtype);
        }
        let [ho, wo] = geom.out_hw();
        let pixels = ho * wo;
        // Columns are stored in vector runs: a run never straddles `cout`.
        if groups != 1 || !geom.cout.is_multiple_of(8) || pixels == 0 || x.dims[0] == 0 {
            return Err(Fallback::Shape);
        }
        if !geom.cin.is_multiple_of(16) {
            return Err(Fallback::Shape);
        }
        let (batches, m) = conv_grid(x, geom);
        candidates(planner.conv_ranked(batches, m, geom), Fallback::Config)
    })
}

/// A convolution's grid batches and output pixels per batch: a static batch
/// folds into the GEMM rows, a bound one walks grid z.
pub fn conv_grid(x: &Extent, geom: &ConvGeom) -> (usize, usize) {
    let [ho, wo] = geom.out_hw();
    if x.var { (x.dims[0], ho * wo) } else { (1, x.dims[0] * ho * wo) }
}

/// `q [b, t, h, d]` against `k`/`v [b, tk, h_kv, d]`, `causal` under that mask,
/// `bias` with an additive bias.
pub fn attention(
    planner: Option<&Planner>,
    dtypes: &[DType],
    q: Option<&Extent>,
    k: Option<&Extent>,
    causal: bool,
    bias: bool,
) -> Plan<FaCfg> {
    plan(|| {
        let planner = gate(planner, dtypes)?;
        let (q, k) = (q.ok_or(Fallback::Symbolic)?, k.ok_or(Fallback::Symbolic)?);
        if q.dims.contains(&0) || k.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        let [b, t, heads, d] = [q.dims[0], q.dims[1], q.dims[2], q.dims[3]];
        candidates(planner.attention_ranked(b * heads, t, k.dims[1], d, causal, bias), Fallback::Shape)
    })
}

/// `qkv [b, t, slots·d]` into heads of width `d`.
pub fn heads(planner: Option<&Planner>, dtypes: &[DType], x: Option<&Extent>, d: usize) -> Plan<NormCfg> {
    plan(|| {
        gate(planner, dtypes)?;
        let x = x.ok_or(Fallback::Symbolic)?;
        if x.dims.len() != 3 || x.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        candidates(config::heads_candidates(d).into(), Fallback::Shape)
    })
}

/// A norm over the last dim of `x`.
pub fn norm(planner: Option<&Planner>, dtypes: &[DType], x: Option<&Extent>) -> Plan<NormCfg> {
    plan(|| {
        gate(planner, dtypes)?;
        let x = x.ok_or(Fallback::Symbolic)?;
        let d = reduced(x)?;
        if x.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        candidates(config::norm_candidates(d).into(), Fallback::Shape)
    })
}

/// `x [n, d]` against the candidates `c [m, d]`, each row's cheapest by
/// `metric` above `floors` (under `groups`); `dtypes` are those of `x` and
/// `c`: 16-bit floats, or 8-bit integers on a target with an 8-bit core
/// (rows scaled per row, [`crate::kernels::nearest::nearest`]). Any depth runs,
/// padded with zeros to the matrix core's ([`Planner::nearest_depth`]), and
/// rows bound to a variable, planned at its bound (as is a list's `k`); bound
/// candidates are not, nor are more than the f32 index holds.
#[allow(clippy::too_many_arguments)]
pub fn nearest(
    planner: Option<&Planner>,
    dtypes: &[DType],
    x: Option<&Extent>,
    c: Option<&Extent>,
    metric: Metric,
    floors: Floors,
    groups: bool,
    k: usize,
) -> Plan<NearestCfg> {
    plan(|| {
        let int8 = dtypes.iter().all(|d| d.scalar() == Some(ScalarDType::Int8));
        let planner = match int8 {
            true => planner
                .filter(|p| config::has_kernels(&p.device.target))
                .ok_or(Fallback::Target)?
                .device
                .target
                .geometry_of(ScalarDType::Int8)
                .and(planner)
                .ok_or(Fallback::Dtype)?,
            false => gate(planner, dtypes)?,
        };
        let dtype = dtypes[0].scalar().ok_or(Fallback::Dtype)?;
        let (x, c) = (x.ok_or(Fallback::Symbolic)?, c.ok_or(Fallback::Symbolic)?);
        if c.var {
            return Err(Fallback::Symbolic);
        }
        let ([n, d], [m, _]) = (dims2(x)?, dims2(c)?);
        if n * m * d * k == 0 || m > MAX_CANDIDATES {
            return Err(Fallback::Shape);
        }
        let d = planner.nearest_depth(d, dtype).ok_or(Fallback::Target)?;
        let p = config::NearestProblem { n, m, d, dtype, metric, floors, groups, k };
        candidates(planner.nearest_ranked(p), Fallback::Config)
    })
}

fn dims2(x: &Extent) -> Result<[usize; 2], Fallback> {
    x.dims[..].try_into().map_err(|_| Fallback::Shape)
}
