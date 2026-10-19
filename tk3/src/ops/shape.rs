//! Which path an op takes, as pure functions of the operands' dims at
//! capacity, the bound batch variable's presence, the dtypes and the target.

use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::{ConstValue, Op, SInt, UOp, ops};
use svod_tensor::Tensor;

use super::config;
use crate::atoms::Target;
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::GemmCfg;
use crate::kernels::rows::NormCfg;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan<C> {
    /// The tile config candidates, the untuned pick first.
    Kernel(Vec<C>),
    Graph(Fallback),
}

/// Why the graph op runs instead of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// No tk3 target for the device.
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

/// A leading dim bound to a runtime variable (the JIT `batch_var`).
#[derive(Clone, Debug)]
pub struct BatchVar {
    /// The dim as the operand spells it, to shrink an output back to.
    pub dim: SInt,
    pub name: String,
    pub min: i64,
    pub max: i64,
}

/// `t`'s dims at capacity and its batch variable; `None` when a dim other
/// than a bound leading one is symbolic.
pub fn extent(shape: &[SInt]) -> Option<(Extent, Option<BatchVar>)> {
    let var = match shape.first() {
        Some(SInt::Symbolic(dim)) => Some(batch_var(dim)?),
        _ => None,
    };
    let dims: Option<Vec<usize>> =
        shape.iter().enumerate().map(|(i, d)| if i == 0 { d.vmax() } else { d.as_const() }).collect();
    let var = var.map(|(name, min, max)| BatchVar { dim: shape[0].clone(), name, min, max });
    Some((Extent { dims: dims?, var: var.is_some() }, var))
}

fn batch_var(dim: &Arc<UOp>) -> Option<(String, i64, i64)> {
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
        Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => Some((name.clone(), *min_val, *max_val)),
        Op::Param(ops::Param { arg, .. }) => {
            let (min, max) = arg.vmin_vmax.as_ref()?;
            Some((arg.name.clone()?, int(&min.0)?, int(&max.0)?))
        }
        _ => None,
    }
}

pub fn shape_of(t: &Tensor) -> svod_tensor::error::Result<Vec<SInt>> {
    Ok(t.shape()?.to_vec())
}

/// The target when it has kernel tables and every operand shares a 16-bit
/// type it has a matrix core for (f32 keeps the graph by policy).
fn gate<'a>(target: Option<&'a Target>, dtypes: &[DType]) -> Result<&'a Target, Fallback> {
    let target = target.filter(|t| config::has_tables(t)).ok_or(Fallback::Target)?;
    let first = dtypes[0].scalar().ok_or(Fallback::Dtype)?;
    let typed = matches!(first, ScalarDType::BFloat16 | ScalarDType::Float16)
        && target.mma(first, ScalarDType::Float32).is_some()
        && dtypes.iter().all(|d| d.scalar() == Some(first));
    typed.then_some(target).ok_or(Fallback::Dtype)
}

/// The last dim, which a bound batch variable must not be.
fn reduced(x: &Extent) -> Result<usize, Fallback> {
    match x.dims[..] {
        [] => Err(Fallback::Shape),
        [_] if x.var => Err(Fallback::Symbolic),
        [.., last] => Ok(last),
    }
}

fn plan<C>(f: impl FnOnce() -> Result<Vec<C>, Fallback>) -> Plan<C> {
    f().map_or_else(Plan::Graph, Plan::Kernel)
}

fn candidates<C>(list: Vec<C>, none: Fallback) -> Result<Vec<C>, Fallback> {
    if list.is_empty() { Err(none) } else { Ok(list) }
}

/// `x [lead..., k] · w [n·halves, k]ᵀ`; `dtypes` are those of every operand.
pub fn linear(target: Option<&Target>, dtypes: &[DType], x: Option<&Extent>, n: usize, gated: bool) -> Plan<GemmCfg> {
    plan(|| {
        let (target, x) = (gate(target, dtypes)?, x.ok_or(Fallback::Symbolic)?);
        let k = reduced(x)?;
        // A bound batch walks grid z; each batch is a GEMM over the rows behind it.
        let lead = &x.dims[..x.dims.len() - 1];
        let (batches, rows) = if x.var { (lead[0], lead[1..].iter().product()) } else { (1, lead.iter().product()) };
        // Columns are stored in vector runs: a run never straddles `n`.
        if !n.is_multiple_of(8) || rows * batches == 0 {
            return Err(Fallback::Shape);
        }
        candidates(config::gemm_candidates(target, batches, rows, n, k, gated), Fallback::Config)
    })
}

/// `q [b, t, h, d]` against `k`/`v [b, tk, h_kv, d]`.
pub fn attention(target: Option<&Target>, dtypes: &[DType], q: Option<&Extent>, k: Option<&Extent>) -> Plan<FaCfg> {
    plan(|| {
        let target = gate(target, dtypes)?;
        let (q, k) = (q.ok_or(Fallback::Symbolic)?, k.ok_or(Fallback::Symbolic)?);
        if q.dims.contains(&0) || k.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        candidates(config::attention_candidates(target, q.dims[3], q.dims[1]), Fallback::Shape)
    })
}

/// `qkv [b, t, slots·d]` into heads of width `d`.
pub fn heads(target: Option<&Target>, dtypes: &[DType], x: Option<&Extent>, d: usize) -> Plan<NormCfg> {
    plan(|| {
        let (target, x) = (gate(target, dtypes)?, x.ok_or(Fallback::Symbolic)?);
        if x.dims.len() != 3 || x.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        candidates(config::heads_candidates(target, d), Fallback::Shape)
    })
}

/// A norm over the last dim of `x`.
pub fn norm(target: Option<&Target>, dtypes: &[DType], x: Option<&Extent>) -> Plan<NormCfg> {
    plan(|| {
        let (target, x) = (gate(target, dtypes)?, x.ok_or(Fallback::Symbolic)?);
        let d = reduced(x)?;
        if x.dims.contains(&0) {
            return Err(Fallback::Shape);
        }
        candidates(config::norm_candidates(target, d), Fallback::Shape)
    })
}
